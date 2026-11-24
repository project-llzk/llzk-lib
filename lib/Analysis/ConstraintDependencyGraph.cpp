//===-- ConstraintDependencyGraph.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/ConstraintDependencyGraph.h"

#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Hash.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/Support/Debug.h>

#include <algorithm>
#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-cdg"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace constrain;
using namespace function;
using namespace pod;

namespace {

/// Infer the half-open index bounds of a statically bounded structured loop.
static std::optional<std::pair<DynamicAPInt, DynamicAPInt>> getStaticLoopIndexRange(Value index) {
  if (auto toIndex = index.getDefiningOp<cast::FeltToIndexOp>()) {
    index = toIndex.getValue();
  }
  auto blockArg = llvm::dyn_cast<BlockArgument>(index);
  if (!blockArg) {
    return std::nullopt;
  }

  if (auto forOp = llvm::dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
    if (blockArg != forOp.getInductionVar()) {
      return std::nullopt;
    }
    auto lower = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto upper = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if (lower && upper) {
      return std::pair(DynamicAPInt(lower.value()), DynamicAPInt(upper.value()));
    }
    return std::nullopt;
  }

  auto whileOp = llvm::dyn_cast<scf::WhileOp>(blockArg.getOwner()->getParentOp());
  if (!whileOp || blockArg.getOwner() != &whileOp.getAfter().front()) {
    return std::nullopt;
  }
  unsigned argumentNumber = blockArg.getArgNumber();
  if (argumentNumber >= whileOp.getBeforeArguments().size() ||
      argumentNumber >= whileOp.getInits().size()) {
    return std::nullopt;
  }

  auto lower = whileOp.getInits()[argumentNumber].getDefiningOp<felt::FeltConstantOp>();
  auto cmp = whileOp.getConditionOp().getCondition().getDefiningOp<boolean::CmpOp>();
  Value beforeArgument = whileOp.getBeforeArguments()[argumentNumber];
  if (!lower || !cmp || cmp.getPredicate() != boolean::FeltCmpPredicate::LT ||
      cmp.getLhs() != beforeArgument ||
      whileOp.getConditionOp().getArgs()[argumentNumber] != beforeArgument) {
    return std::nullopt;
  }
  auto upper = cmp.getRhs().getDefiningOp<felt::FeltConstantOp>();
  if (!upper) {
    return std::nullopt;
  }
  auto upperBound = upper.getValue().getReducedValue();
  if (!upperBound) {
    // A fieldless literal is not a bound until its field representative is known.
    return std::nullopt;
  }
  // The backedge need not be monotone. The condition proves an upper bound, while a felt's
  // representative is nonnegative regardless of the initializer or recurrence.
  return std::pair(DynamicAPInt(0), *upperBound);
}

/// Select constant indices, known loop ranges, or whole dimensions from operand states.
static std::vector<SourceRefIndex> getArrayAccessIndices(
    ArrayAccessOpInterface access,
    llvm::function_ref<const SourceRefLatticeValue &(Value)> getIndexState
) {
  std::vector<SourceRefIndex> indices;
  ArrayType arrayType = access.getArrRef().getType();
  for (auto [i, index] : llvm::enumerate(access.getIndices())) {
    int64_t dimSize = arrayType.getDimSize(llzk::checkedCast<unsigned>(i));
    auto constantIndex = getIndexState(index).getConstantValue();
    if (succeeded(constantIndex)) {
      indices.emplace_back(*constantIndex);
    } else if (
        auto bounds = getStaticLoopIndexRange(index); bounds && !ShapedType::isDynamic(dimSize)
    ) {
      // A guarded access may be valid even when the loop exceeds the array's bounds.
      auto lower = std::clamp(bounds->first, DynamicAPInt(0), DynamicAPInt(dimSize));
      auto upper = std::clamp(bounds->second, lower, DynamicAPInt(dimSize));
      indices.emplace_back(lower, upper);
    } else {
      indices.push_back(SourceRefIndex::forArrayDimension(dimSize));
    }
  }
  return indices;
}

/// Extract array selections in the dimension order used by SourceRef::narrowRanges.
static SmallVector<SourceRefIndex> getReferenceArrayIndices(const SourceRef &ref) {
  SmallVector<SourceRefIndex> indices;
  llvm::copy_if(ref.getPath(), std::back_inserter(indices), [](const SourceRefIndex &index) {
    return index.isIndex() || index.isIndexRange();
  });
  return indices;
}

/// Match a descendant's prefix by overlap, allowing concrete indices inside queried ranges.
static bool hasOverlappingPrefix(const SourceRef &candidate, const SourceRef &prefix) {
  auto ancestor = FailureOr<SourceRef>(candidate);
  while (succeeded(ancestor) && ancestor->getPath().size() > prefix.getPath().size()) {
    ancestor = ancestor->getParentPrefix();
  }
  return succeeded(ancestor) && ancestor->overlaps(prefix);
}

/// Prove correspondence only for single-source reads of the same SSA indices. Matching ranges
/// alone do not prove correspondence: independent loop variables can have identical bounds.
static bool haveCorrespondingArrayIndices(Operation *emitOp, DataFlowSolver &solver) {
  if (!llvm::isa<EmitEqualityOp>(emitOp) || emitOp->getNumOperands() != 2) {
    return false;
  }
  auto lhs = emitOp->getOperand(0).getDefiningOp<ReadArrayOp>();
  auto rhs = emitOp->getOperand(1).getDefiningOp<ReadArrayOp>();
  if (!lhs || !rhs || !llvm::equal(lhs.getIndices(), rhs.getIndices())) {
    return false;
  }
  for (ReadArrayOp read : {lhs, rhs}) {
    auto value = SourceRefAnalysis::getValueState(solver, read.getResult());
    if (!value.isSingleValue()) {
      return false;
    }
    llvm::SmallDenseMap<Value, SourceRefLatticeValue> indexStates;
    for (Value index : read.getIndices()) {
      indexStates.try_emplace(index, SourceRefAnalysis::getValueState(solver, index));
    }
    auto indices =
        getArrayAccessIndices(read, [&indexStates](Value index) -> const SourceRefLatticeValue & {
      return indexStates.find(index)->second;
    });
    if (!llvm::equal(indices, getReferenceArrayIndices(value.getSingleValue()))) {
      return false;
    }
  }
  return true;
}

/// Recognize the field-independent literals zero and one, including fieldless constants.
static bool isBooleanLiteral(const SourceRef &ref, int64_t literal) {
  if (auto value = ref.getConstant(); succeeded(value)) {
    if (auto constant = value->getDefiningOp<felt::FeltConstantOp>()) {
      return constant.getValue().getReducedValueOrRaw() == literal;
    }
  }
  auto index = ref.getConstantIndexValue();
  return succeeded(index) && *index == literal;
}

/// Collect POD record positions used to distinguish indexed aggregate alternatives.
static llvm::SmallVector<std::pair<uint64_t, uint64_t>>
getAggregateAlternativeOrdinals(const SourceRef &ref) {
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> result;
  auto root = ref.getRoot();
  if (failed(root)) {
    return result;
  }
  Type currentType = root->getType();
  for (const SourceRefIndex &index : ref.getPath()) {
    if (index.isMember()) {
      currentType = index.getMember().getType();
      continue;
    }
    if (index.isPodRecord()) {
      auto podType = llvm::dyn_cast<PodType>(currentType);
      if (!podType) {
        continue;
      }
      auto records = podType.getRecords();
      for (auto [ordinal, record] : llvm::enumerate(records)) {
        if (record.getName() == index.getPodRecordNameAttr()) {
          result.emplace_back(ordinal, records.size());
          currentType = record.getType();
          break;
        }
      }
      continue;
    }
    if (index.isIndex() || index.isIndexRange()) {
      if (auto arrayType = llvm::dyn_cast<ArrayType>(currentType)) {
        currentType = arrayType.getElementType();
      }
    }
  }
  return result;
}

/// Check whether a POD alternative matches the selected aggregate element.
static bool isCompatibleIndexedAlternative(const SourceRef &candidate, const SourceRef &selection) {
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> selectedIndices;
  auto root = selection.getRoot();
  if (failed(root)) {
    return true;
  }
  Type currentType = root->getType();
  for (const SourceRefIndex &index : selection.getPath()) {
    if (index.isMember()) {
      currentType = index.getMember().getType();
      continue;
    }
    if (index.isPodRecord()) {
      if (auto podType = llvm::dyn_cast<PodType>(currentType)) {
        for (auto record : podType.getRecords()) {
          if (record.getName() == index.getPodRecordNameAttr()) {
            currentType = record.getType();
            break;
          }
        }
      }
      continue;
    }
    if (auto arrayType = llvm::dyn_cast<ArrayType>(currentType)) {
      if (index.isIndex()) {
        selectedIndices.emplace_back(
            static_cast<uint64_t>(static_cast<int64_t>(index.getIndex())), arrayType.getDimSize(0)
        );
      }
      currentType = arrayType.getElementType();
    }
  }

  for (auto [ordinal, alternativeCount] : getAggregateAlternativeOrdinals(candidate)) {
    for (auto [selected, dimensionSize] : selectedIndices) {
      if (alternativeCount == dimensionSize && ordinal != selected) {
        return false;
      }
    }
  }
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// SourceRefAnalysis
//===----------------------------------------------------------------------===//

const SourceRefAnalysis::Lattice *SourceRefAnalysis::getLattice(DataFlowSolver &solver, Value val) {
  return solver.lookupState<Lattice>(val);
}

SourceRefLatticeValue SourceRefAnalysis::getValueState(DataFlowSolver &solver, Value val) {
  if (const auto *state = getLattice(solver, val)) {
    return state->getValue();
  }
  return SourceRefLattice::getDefaultValue(val);
}

mlir::FailureOr<SourceRefLatticeValue>
SourceRefAnalysis::getWriteTargetState(DataFlowSolver &solver, Operation *op) {
  llvm::SmallDenseMap<Value, SourceRefLatticeValue, 4> operandVals;
  for (Value operand : op->getOperands()) {
    operandVals[operand] = getValueState(solver, operand);
  }

  SymbolTableCollection tables;
  if (auto memberRefOp = llvm::dyn_cast<MemberRefOpInterface>(op)) {
    if (!memberRefOp.isRead()) {
      auto memberOpRes = memberRefOp.getMemberDefOp(tables);
      ensure(succeeded(memberOpRes), "could not find member write");
      auto componentIt = operandVals.find(memberRefOp.getComponent());
      ensure(componentIt != operandVals.end(), "missing component lattice for member write");
      auto memberValsRes = componentIt->second.referenceMember(memberOpRes.value());
      ensure(succeeded(memberValsRes), "could not create SourceRef child for member write");
      return memberValsRes->first;
    }
  }

  if (auto podAccessOp = llvm::dyn_cast<PodAccessOpInterface>(op)) {
    if (!podAccessOp.isRead()) {
      auto podIt = operandVals.find(podAccessOp.getPodRef());
      ensure(podIt != operandVals.end(), "missing pod lattice for pod write");
      auto podValsRes = podIt->second.referencePodRecord(podAccessOp.getRecordNameAttr());
      ensure(succeeded(podValsRes), "could not create SourceRef child for pod write");
      return podValsRes->first;
    }
  }

  if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    if (llvm::isa<WriteArrayOp, InsertArrayOp>(arrayAccessOp)) {
      auto array = arrayAccessOp.getArrRef();
      auto it = operandVals.find(array);
      ensure(it != operandVals.end(), "improperly constructed operandVals map");
      const auto &currVals = it->second;

      auto indices =
          getArrayAccessIndices(arrayAccessOp, [&](Value index) -> const SourceRefLatticeValue & {
        auto idxIt = operandVals.find(index);
        ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
        return idxIt->second;
      });

      auto newValsRes = currVals.extract(indices);
      ensure(succeeded(newValsRes), "could not create SourceRef child for array access");
      auto [newVals, _] = *newValsRes;
      if (llvm::isa<WriteArrayOp>(arrayAccessOp)) {
        ensure(newVals.isScalar(), "array write must produce a scalar value");
      }
      return newVals;
    }
  }

  return mlir::failure();
}

void SourceRefAnalysis::setToEntryState(Lattice *lattice) {
  if (Value value = lattice->getAnchor()) {
    if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (llvm::isa_and_present<RegionBranchOpInterface>(parent) &&
          llvm::isa<ArrayType, StructType, PodType>(value.getType())) {
        // Region-branch arguments are aliases of their incoming aggregate storage. Giving them a
        // fresh root would make loop-carried writes unstable and would discard that identity.
        (void)lattice->setValue(SourceRefLatticeValue());
        return;
      }
    }
    (void)lattice->setValue(SourceRefLattice::getDefaultValue(value));
  }
}

LogicalResult SourceRefAnalysis::visitOperation(
    Operation *op, ArrayRef<const Lattice *> operands, ArrayRef<Lattice *> results
) {
  LLVM_DEBUG(llvm::dbgs() << "SourceRefAnalysis::visitOperation: " << *op << '\n');

  DenseMap<Value, const Lattice *> operandVals;
  for (auto [operand, lattice] : llvm::zip(op->getOperands(), operands)) {
    operandVals[operand] = lattice;
  }

  if (auto memberRefOp = llvm::dyn_cast<MemberRefOpInterface>(op)) {
    auto memberOpRes = memberRefOp.getMemberDefOp(tables);
    ensure(succeeded(memberOpRes), "could not find member read");
    auto memberValsRes =
        operandVals.at(memberRefOp.getComponent())->getValue().referenceMember(memberOpRes.value());
    ensure(succeeded(memberValsRes), "could not create SourceRef child for member reference");
    if (memberRefOp.isRead()) {
      auto [memberVals, _] = *memberValsRes;
      propagateIfChanged(results.front(), results.front()->setValue(memberVals));
    }
    return success();
  }

  if (auto podAccessOp = llvm::dyn_cast<PodAccessOpInterface>(op)) {
    auto podValsRes = operandVals.at(podAccessOp.getPodRef())
                          ->getValue()
                          .referencePodRecord(podAccessOp.getRecordNameAttr());
    ensure(succeeded(podValsRes), "could not create SourceRef child for pod reference");
    if (podAccessOp.isRead()) {
      auto [podVals, _] = *podValsRes;
      propagateIfChanged(results.front(), results.front()->setValue(podVals));
    }
    return success();
  }

  if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    if (!results.empty()) {
      auto newVals = arraySubdivisionOpUpdate(arrayAccessOp, operandVals);
      propagateIfChanged(results.front(), results.front()->setValue(newVals));
    }
    return success();
  }

  if (auto createArray = llvm::dyn_cast<CreateArrayOp>(op)) {
    auto createArrayRes = createArray.getResult();
    const auto &elements = createArray.getElements();
    if (elements.empty()) {
      propagateIfChanged(
          results.front(),
          results.front()->setValue(SourceRef(llvm::cast<OpResult>(createArrayRes)))
      );
      return success();
    }

    SourceRefLatticeValue newArrayVal(createArray.getType().getShape());
    for (auto [i, element] : llvm::enumerate(elements)) {
      (void)newArrayVal.getElemFlatIdx(i).setValue(operandVals.at(element)->getValue());
    }
    propagateIfChanged(results.front(), results.front()->setValue(newArrayVal));
    return success();
  }

  if (auto newPod = llvm::dyn_cast<NewPodOp>(op)) {
    auto newPodValue = SourceRefLattice::getDefaultValue(newPod.getResult());
    propagateIfChanged(results.front(), results.front()->setValue(newPodValue));
    return success();
  }

  if (auto structNewOp = llvm::dyn_cast<CreateStructOp>(op)) {
    auto newStructValue = SourceRefLattice::getDefaultValue(structNewOp.getResult());
    propagateIfChanged(results.front(), results.front()->setValue(newStructValue));
    return success();
  }

  auto updated = fallbackOpUpdate(op, operandVals, results);
  for (Lattice *result : results) {
    propagateIfChanged(result, updated);
  }
  return success();
}

/// Return whether a value can carry shared component identity, including through value aggregates.
static bool containsComponentHandle(Type type) {
  // An unresolved type variable may be instantiated with a component handle or an aggregate
  // containing one, so its storage effects must also be treated conservatively.
  if (llvm::isa<StructType, polymorphic::TypeVarType>(type)) {
    return true;
  }
  if (auto array = llvm::dyn_cast<ArrayType>(type)) {
    return containsComponentHandle(array.getElementType());
  }
  if (auto pod = llvm::dyn_cast<PodType>(type)) {
    return llvm::any_of(pod.getRecords(), [](auto record) {
      return containsComponentHandle(record.getType());
    });
  }
  return false;
}

LogicalResult SourceRefAnalysis::validateInput(ModuleOp module) {
  auto result = module.walk([](Operation *op) -> WalkResult {
    auto call = llvm::dyn_cast<CallOpInterface>(op);
    if (!call || llvm::none_of(call.getArgOperands().getTypes(), containsComponentHandle)) {
      return WalkResult::advance();
    }
    auto callable = llvm::dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
    if (callable && callable.getCallableRegion()) {
      return WalkResult::advance();
    }
    return op->emitOpError(
        "cannot analyze an opaque call with component handles in its arguments; "
        "callee storage effects are unknown"
    );
  });
  return failure(result.wasInterrupted());
}

void SourceRefAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const Lattice *> operandLattices,
    ArrayRef<Lattice *> resultLattices
) {
  auto callable = dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  if (!callable || !callable.getCallableRegion()) {
    // Call is truly external
    for (auto [result, lattice] : llvm::zip(call->getResults(), resultLattices)) {
      auto resultRef = SourceRefLattice::getSourceRef(result);
      ensure(succeeded(resultRef), "could not create external call SourceRef");
      propagateIfChanged(lattice, lattice->setValue(*resultRef));
    }
    return;
  }
  if (resultLattices.empty()) {
    // `verif.include` and other no-result call-like ops still need to be
    // treated as valid callable edges, but there are no results to
    // translate back to the caller.
    return;
  }
  // Call is to a defined function with a body, but it's treated as external so we
  // can translate the results based on the arguments.
  auto funcOpRes = resolveCallable<FuncDefOp>(tables, call);
  ensure(succeeded(funcOpRes), "could not lookup called function");
  auto funcOp = funcOpRes->get();

  const auto *predecessors = getOrCreateFor<mlir::dataflow::PredecessorState>(
      getProgramPointAfter(call), getProgramPointAfter(call)
  );
  // If not all return sites are known, then conservatively assume we can't
  // reason about the data-flow.
  if (!predecessors->allPredecessorsKnown()) {
    setAllToEntryStates(resultLattices);
    return;
  }
  const auto returnSites = predecessors->getKnownPredecessors();

  std::unordered_map<SourceRef, SourceRefLatticeValue, SourceRef::Hash> translation;
  for (auto [arg, lattice] : llvm::zip_equal(funcOp.getArguments(), operandLattices)) {
    translation[SourceRef(arg)] = lattice->getValue();
  }

  for (auto [result, resultLattice] : llvm::zip(call->getResults(), resultLattices)) {
    SourceRefLatticeValue combined;
    unsigned resultNum = result.getResultNumber();
    for (Operation *returnSite : returnSites) {
      auto retVal = static_cast<const Lattice *>(getLatticeElementFor(
                                                     getProgramPointAfter(call.getOperation()),
                                                     returnSite->getOperand(resultNum)
                                                 ))
                        ->getValue();
      auto [translatedVal, _] = retVal.translate(translation);
      (void)combined.update(translatedVal);
    }
    propagateIfChanged(resultLattice, resultLattice->setValue(combined));
  }
}

ChangeResult SourceRefAnalysis::fallbackOpUpdate(
    Operation *op, const OperandValues &operandVals, ArrayRef<Lattice *> results
) {
  auto updated = ChangeResult::NoChange;
  for (auto [res, lattice] : llvm::zip(op->getResults(), results)) {
    auto cur = SourceRefLattice::getDefaultValue(res);
    for (const auto &[_, opVal] : operandVals) {
      (void)cur.update(opVal->getValue());
    }
    updated |= lattice->setValue(cur);
  }
  return updated;
}

SourceRefLatticeValue SourceRefAnalysis::arraySubdivisionOpUpdate(
    ArrayAccessOpInterface arrayAccessOp, const OperandValues &operandVals
) {
  auto array = arrayAccessOp.getArrRef();
  auto it = operandVals.find(array);
  ensure(it != operandVals.end(), "improperly constructed operandVals map");
  const auto &currVals = it->second->getValue();

  auto indices =
      getArrayAccessIndices(arrayAccessOp, [&](Value index) -> const SourceRefLatticeValue & {
    auto idxIt = operandVals.find(index);
    ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
    return idxIt->second->getValue();
  });

  auto newValsRes = currVals.extract(indices);
  ensure(succeeded(newValsRes), "could not create SourceRef child for array access");
  auto [newVals, _] = *newValsRes;
  if (llvm::isa<ReadArrayOp, WriteArrayOp>(arrayAccessOp)) {
    ensure(newVals.isScalar(), "array read/write must produce a scalar value");
  }
  return newVals;
}

//===----------------------------------------------------------------------===//
// ConstraintDependencyGraph
//===----------------------------------------------------------------------===//

FailureOr<ConstraintDependencyGraph> ConstraintDependencyGraph::compute(
    ModuleOp m, StructDefOp s, DataFlowSolver &solver, AnalysisManager &am,
    const CDGAnalysisContext &ctx
) {
  ConstraintDependencyGraph cdg(m, s, ctx);
  if (cdg.computeConstraints(solver, am).failed()) {
    return mlir::failure();
  }
  return cdg;
}

void ConstraintDependencyGraph::dump() const { print(llvm::errs()); }

/// Print all constraints. Any element that is unconstrained is omitted.
void ConstraintDependencyGraph::print(llvm::raw_ostream &os) const {
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them.
  // We also want to add the constant values into the printing.
  std::set<std::set<SourceRef>> sortedSets;
  for (const auto *entry : signalSets) {
    if (!entry->isLeader()) {
      continue;
    }

    auto members = signalSets.members(*entry);
    std::set<SourceRef> sortedMembers(members.begin(), members.end());

    // We only want to print sets with a size > 1, because size == 1 means the
    // signal is not in a constraint.
    if (sortedMembers.size() > 1) {
      sortedSets.insert(sortedMembers);
    }
  }
  // Add the constants in separately.
  for (const auto &[ref, constSet] : constantSets) {
    if (constSet.empty()) {
      continue;
    }
    std::set<SourceRef> sortedMembers(constSet.begin(), constSet.end());
    sortedMembers.insert(ref);
    sortedSets.insert(sortedMembers);
  }

  os << "ConstraintDependencyGraph { ";

  llvm::interleave(sortedSets, os, [&](const auto &members) {
    os << "\n    { ";
    llvm::interleaveComma(members, os);
    os << " }";
  }, ",");
  if (!sortedSets.empty()) {
    os << '\n';
  }

  os << "}\n";
}

mlir::LogicalResult ConstraintDependencyGraph::computeConstraints(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  FuncDefOp constrainFnOp = structDef.getConstrainFuncOp();
  ensure(
      constrainFnOp,
      "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
  );

  /**
   * Now, given the analysis, construct the CDG:
   * - Union all references based on solver results.
   * - Union all references based on nested dependencies.
   */

  // - Union all constraints from the analysis
  // This requires iterating over all of the emit operations
  constrainFnOp.walk([this, &solver](Operation *op) {
    if (!dataflow::isOperationLive(solver, op)) {
      return;
    }

    for (Value operand : op->getOperands()) {
      auto operandRefs = SourceRefAnalysis::getValueState(solver, operand).foldToScalar();
      for (const SourceRef &ref : operandRefs) {
        ref2Val[ref].insert(operand);
      }
    }
    for (Value result : op->getResults()) {
      auto resultRefs = SourceRefAnalysis::getValueState(solver, result).foldToScalar();
      for (const SourceRef &ref : resultRefs) {
        ref2Val[ref].insert(result);
      }
    }
    auto writeTargetState = SourceRefAnalysis::getWriteTargetState(solver, op);
    if (succeeded(writeTargetState)) {
      for (const SourceRef &ref : writeTargetState->foldToScalar()) {
        ref2Val[ref].insert(op);
      }
    }
    if (isa<EmitEqualityOp, EmitContainmentOp>(op)) {
      this->walkConstrainOp(solver, op);
    }
  });

  /**
   * Step two of the analysis is to traverse all of the constrain calls.
   * This is the nested analysis, basically.
   * Constrain functions don't return, so we don't need to compute "values" from
   * the call. We just need to see what constraints are generated here, and
   * add them to the transitive closures.
   */
  auto fnCallWalker = [this, &solver, &am](CallOp fnCall) mutable {
    if (!dataflow::isOperationLive(solver, fnCall.getOperation())) {
      return;
    }
    auto res = resolveCallable<FuncDefOp>(tables, fnCall);
    ensure(mlir::succeeded(res), "could not resolve constrain call");

    auto fn = res->get();
    if (!fn.isStructConstrain()) {
      return;
    }
    // Nested
    auto calledStruct = fn.getOperation()->getParentOfType<StructDefOp>();
    SourceRefRemappings translations;

    // Map fn parameters to args in the call op
    for (auto [arg, operand] : llvm::zip_equal(fn.getArguments(), fnCall.getArgOperands())) {
      translations.emplace_back(SourceRef(arg), SourceRefAnalysis::getValueState(solver, operand));
    }
    auto &childAnalysis =
        am.getChildAnalysis<ConstraintDependencyGraphStructAnalysis>(calledStruct);
    if (!childAnalysis.constructed(ctx)) {
      ensure(
          mlir::succeeded(childAnalysis.runAnalysis(solver, am, {.runIntraprocedural = false})),
          "could not construct CDG for child struct"
      );
    }
    auto translatedCDG = childAnalysis.getResult(ctx).translate(translations);
    // Update the refMap with the translation
    const auto &translatedRef2Val = translatedCDG.getRef2Val();
    ref2Val.insert(translatedRef2Val.begin(), translatedRef2Val.end());

    // Now, union sets based on the translation
    // We should be able to just merge what is in the translatedCDG to the current CDG
    auto &tSets = translatedCDG.signalSets;
    for (const auto *entry : tSets) {
      if (!entry->isLeader()) {
        continue;
      }
      auto leader = entry->getData();
      for (const SourceRef &member : tSets.members(*entry)) {
        signalSets.unionSets(leader, member);
      }
    }
    for (const auto &[source, targets] : translatedCDG.dependencyEdges) {
      for (const auto &[target, elementwise] : targets) {
        addDependencyEdge(source, target, elementwise);
      }
    }
    for (const auto &[source, targets] : translatedCDG.conditionalDependencyEdges) {
      conditionalDependencyEdges[source].insert(targets.begin(), targets.end());
    }
    auto &translatedAliases = translatedCDG.directAliases;
    for (const auto *leaderIt : translatedAliases) {
      if (!leaderIt->isLeader()) {
        continue;
      }
      const SourceRef &leader = leaderIt->getData();
      for (const SourceRef &member : translatedAliases.members(*leaderIt)) {
        directAliases.unionSets(leader, member);
      }
    }
    // And update the constant sets
    for (auto &[ref, constSet] : translatedCDG.constantSets) {
      constantSets[ref].insert(constSet.begin(), constSet.end());
    }
  };
  if (!ctx.runIntraproceduralAnalysis()) {
    constrainFnOp.walk(fnCallWalker);
  }

  return mlir::success();
}

void ConstraintDependencyGraph::addDependencyEdge(
    const SourceRef &source, const SourceRef &target, bool elementwise
) {
  auto [it, inserted] = dependencyEdges[source].try_emplace(target, elementwise);
  if (!inserted) {
    it->second &= elementwise;
  }
}

void ConstraintDependencyGraph::walkConstrainOp(
    mlir::DataFlowSolver &solver, mlir::Operation *emitOp
) {
  std::vector<SourceRef> signalUsages, constUsages;
  std::vector<SourceRef> conditionalUsages;
  llvm::SmallVector<SourceRefSet, 2> operandSignals;

  // Some frontends preserve a witness-only conditional in @constrain as an empty scf.if marker.
  // Associate that marker's condition with the next emitted constraint in the block.
  for (Operation *previous = emitOp->getPrevNode(); previous != nullptr;
       previous = previous->getPrevNode()) {
    if (llvm::isa<EmitEqualityOp, EmitContainmentOp>(previous)) {
      break;
    }
    auto ifOp = llvm::dyn_cast<scf::IfOp>(previous);
    if (!ifOp) {
      continue;
    }
    auto hasOnlyTerminator = [](Region &region) {
      return region.hasOneBlock() && region.front().without_terminator().empty();
    };
    if (!hasOnlyTerminator(ifOp.getThenRegion()) || !hasOnlyTerminator(ifOp.getElseRegion())) {
      continue;
    }
    for (const SourceRef &conditionRef :
         SourceRefAnalysis::getValueState(solver, ifOp.getCondition()).foldToScalar()) {
      if (conditionRef.isConstant() || conditionRef.isTemplateConstant()) {
        constUsages.push_back(conditionRef);
      } else if (!conditionRef.isBlockArgument() || *conditionRef.getInputNum() == 0) {
        conditionalUsages.push_back(conditionRef);
      }
    }
  }

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = SourceRefAnalysis::getValueState(solver, operand);
    SourceRefSet currentOperandSignals;
    for (const auto &ref : latticeVal.foldToScalar()) {
      if (ref.isConstant() || ref.isTemplateConstant()) {
        constUsages.push_back(ref);
      } else {
        signalUsages.push_back(ref);
      }
    }
    for (const SourceRef &ref : SourceRefAnalysis::getValueState(solver, operand).foldToScalar()) {
      if (!ref.isConstant() && !ref.isTemplateConstant()) {
        currentOperandSignals.insert(ref);
      }
    }
    operandSignals.push_back(std::move(currentOperandSignals));
  }

  auto isStorageIdentity = [](Value operand) {
    return llvm::isa<BlockArgument>(operand) ||
           llvm::isa_and_present<MemberReadOp, ReadArrayOp, ReadPodOp>(operand.getDefiningOp());
  };
  if (llvm::isa<EmitEqualityOp>(emitOp) && operandSignals.size() == 2 &&
      llvm::all_of(emitOp->getOperands(), isStorageIdentity) && operandSignals[0].size() == 1 &&
      operandSignals[1].size() == 1) {
    const SourceRef &lhs = *operandSignals[0].begin();
    const SourceRef &rhs = *operandSignals[1].begin();
    directAliases.unionSets(lhs, rhs);
  }

  // Compute a transitive closure over the signals.
  if (!signalUsages.empty()) {
    const bool elementwise = haveCorrespondingArrayIndices(emitOp, solver);
    for (const SourceRef &lhs : signalUsages) {
      for (const SourceRef &rhs : signalUsages) {
        if (lhs != rhs) {
          addDependencyEdge(lhs, rhs, elementwise);
        }
      }
    }
    auto leader = signalSets.getOrInsertLeaderValue(signalUsages.front());
    for (const SourceRef &signal : llvm::drop_begin(signalUsages)) {
      signalSets.unionSets(leader, signal);
    }
  }
  for (const SourceRef &conditionRef : conditionalUsages) {
    for (const SourceRef &signalRef : signalUsages) {
      if (conditionRef != signalRef) {
        conditionalDependencyEdges[conditionRef].insert(signalRef);
        conditionalDependencyEdges[signalRef].insert(conditionRef);
      }
    }
  }
  // Also update constant references for each value.
  for (auto &sig : signalUsages) {
    constantSets[sig].insert(constUsages.begin(), constUsages.end());
  }
}

ConstraintDependencyGraph
ConstraintDependencyGraph::translate(SourceRefRemappings translation) const {
  ConstraintDependencyGraph res(mod, structDef, ctx);
  auto translate =
      [&translation](const SourceRef &elem) -> mlir::FailureOr<std::vector<SourceRef>> {
    std::vector<SourceRef> refs;
    for (auto &[prefix, vals] : translation) {
      if (!elem.isValidPrefix(prefix)) {
        continue;
      }

      if (vals.isArray()) {
        // Try to index into the array
        auto suffix = elem.getSuffix(prefix);
        ensure(
            mlir::succeeded(suffix), "failure is nonsensical, we already checked for valid prefix"
        );

        auto resolvedValsRes = vals.extract(suffix.value());
        ensure(succeeded(resolvedValsRes), "could not create SourceRef child while resolving refs");
        auto [resolvedVals, _] = *resolvedValsRes;
        auto folded = resolvedVals.foldToScalar();
        refs.insert(refs.end(), folded.begin(), folded.end());
      } else {
        for (const auto &replacement : vals.getScalarValue()) {
          auto translated = elem.translate(prefix, replacement);
          if (mlir::succeeded(translated)) {
            refs.push_back(translated.value());
          }
        }
      }
    }
    if (refs.empty()) {
      return mlir::failure();
    }
    return refs;
  };

  for (const auto *entry : signalSets) {
    if (!entry->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<SourceRef> translatedSignals, translatedConsts;
    for (const SourceRef &member : signalSets.members(*entry)) {
      auto translated = translate(member);
      if (mlir::failed(translated)) {
        continue;
      }
      for (const auto &ref : *translated) {
        if (ref.isConstant()) {
          translatedConsts.push_back(ref);
        } else {
          translatedSignals.push_back(ref);
        }
      }
      // Also add the constants from the original CDG
      if (auto it = constantSets.find(member); it != constantSets.end()) {
        const auto &origConstSet = it->second;
        translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
      }
    }

    if (translatedSignals.empty()) {
      continue;
    }

    // Now we can insert the translated signals
    auto leader = translatedSignals.front();
    res.signalSets.insert(leader);
    for (const SourceRef &signal : llvm::drop_begin(translatedSignals)) {
      res.signalSets.insert(signal);
      res.signalSets.unionSets(leader, signal);
    }

    // And update the constant references
    for (auto &ref : translatedSignals) {
      res.constantSets[ref].insert(translatedConsts.begin(), translatedConsts.end());
    }
  }

  // Contract untranslated vertices, retaining paths between caller-visible endpoints.
  auto translateEdges = [&translate, &res](const auto &edges) {
    for (const auto &[source, targets] : edges) {
      auto translatedSources = translate(source);
      if (failed(translatedSources)) {
        continue;
      }
      SourceRefSet visitedInternal;
      SmallVector<std::pair<SourceRef, bool>> pending;
      for (const auto &[target, elementwise] : targets) {
        pending.emplace_back(target, elementwise);
      }
      while (!pending.empty()) {
        auto [target, elementwise] = pending.pop_back_val();
        auto translatedTargets = translate(target);
        if (failed(translatedTargets)) {
          if (!visitedInternal.insert(target).second) {
            continue;
          }
          if (auto it = edges.find(target); it != edges.end()) {
            for (const auto &[next, _] : it->second) {
              // A contracted path carries connectivity, without assuming index correlation.
              pending.emplace_back(next, false);
            }
          }
          continue;
        }
        for (const SourceRef &translatedSource : *translatedSources) {
          for (const SourceRef &translatedTarget : *translatedTargets) {
            if (!translatedSource.isConstant() && !translatedTarget.isConstant() &&
                translatedSource != translatedTarget) {
              // Prefix translation may add or remove array dimensions. Preserve the proof
              // only when both endpoints retain their original array selections.
              res.addDependencyEdge(
                  translatedSource, translatedTarget,
                  elementwise &&
                      getReferenceArrayIndices(source) ==
                          getReferenceArrayIndices(translatedSource) &&
                      getReferenceArrayIndices(target) == getReferenceArrayIndices(translatedTarget)
              );
            }
          }
        }
      }
    }
  };
  translateEdges(dependencyEdges);

  // Translate conditional edges without treating them as emitted constraints.
  auto translateConditionalEdges = [&translate](const auto &edges, auto &translatedEdges) {
    for (const auto &[source, targets] : edges) {
      auto translatedSources = translate(source);
      if (failed(translatedSources)) {
        continue;
      }
      for (const SourceRef &target : targets) {
        auto translatedTargets = translate(target);
        if (failed(translatedTargets)) {
          continue;
        }
        for (const SourceRef &translatedSource : *translatedSources) {
          for (const SourceRef &translatedTarget : *translatedTargets) {
            if (!translatedSource.isConstant() && !translatedTarget.isConstant() &&
                translatedSource != translatedTarget) {
              translatedEdges[translatedSource].insert(translatedTarget);
            }
          }
        }
      }
    }
  };
  for (const auto *leaderIt : directAliases) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    std::vector<SourceRef> translatedAliases;
    for (auto memberIt = directAliases.member_begin(*leaderIt);
         memberIt != directAliases.member_end(); ++memberIt) {
      auto translated = translate(*memberIt);
      if (failed(translated)) {
        continue;
      }
      llvm::copy_if(
          *translated, std::back_inserter(translatedAliases),
          [](const SourceRef &translatedRef) { return !translatedRef.isConstant(); }
      );
    }
    if (translatedAliases.empty()) {
      continue;
    }
    const SourceRef &leader = translatedAliases.front();
    res.directAliases.insert(leader);
    for (const SourceRef &alias : llvm::drop_begin(translatedAliases)) {
      res.directAliases.unionSets(leader, alias);
    }
  }

  translateConditionalEdges(conditionalDependencyEdges, res.conditionalDependencyEdges);

  // Translate ref2Val as well
  for (const auto &[ref, vals] : ref2Val) {
    auto translationRes = translate(ref);
    if (succeeded(translationRes)) {
      for (const auto &translatedRef : *translationRes) {
        res.ref2Val[translatedRef].insert(vals.begin(), vals.end());
      }
    }
  }

  return res;
}

SourceRefSet ConstraintDependencyGraph::getConstrainingValues(const SourceRef &ref) const {
  SourceRefSet res;
  SourceRefSet visited;
  SmallVector<SourceRef> worklist = {ref};
  bool hasBooleanZero = false;
  bool hasBooleanOne = false;
  for (const auto &[source, constants] : constantSets) {
    if (!source.overlaps(ref)) {
      continue;
    }
    for (const SourceRef &constant : constants) {
      hasBooleanZero |= isBooleanLiteral(constant, 0);
      hasBooleanOne |= isBooleanLiteral(constant, 1);
    }
  }
  const bool followConditionalDependencies = hasBooleanZero && hasBooleanOne;

  // Overlapping aggregate ranges can bridge otherwise distinct equivalence sets. Follow those
  // bridges transitively so that, for example, `out[2] == child.out[2]` and a child constraint on
  // `child.out[0:N]` connect `out[2]` to the child's inputs.
  while (!worklist.empty()) {
    const SourceRef selection = worklist.pop_back_val();
    if (!visited.insert(selection).second) {
      continue;
    }
    const bool includeDescendants = llvm::isa<ArrayType, StructType, PodType>(selection.getType());
    auto currRef = FailureOr<SourceRef>(selection);
    while (succeeded(currRef)) {
      // Visiting a parent for its constraints does not select its other descendants.
      // Keep parents out of visited so a later edge can still select the whole aggregate.
      const bool expandDescendants = includeDescendants && *currRef == selection;

      for (const auto &[source, targets] : dependencyEdges) {
        if ((!source.overlaps(*currRef) &&
             !(expandDescendants && hasOverlappingPrefix(source, *currRef))) ||
            !isCompatibleIndexedAlternative(source, ref)) {
          continue;
        }
        if (auto constIt = constantSets.find(source); constIt != constantSets.end()) {
          res.insert(constIt->second.begin(), constIt->second.end());
        }
        for (const auto &[target, elementwise] : targets) {
          if (!isCompatibleIndexedAlternative(target, ref)) {
            continue;
          }
          SourceRef narrowed = elementwise ? target.narrowRanges(*currRef) : target;
          if (!visited.contains(narrowed)) {
            worklist.push_back(narrowed);
          }
          if (!narrowed.overlaps(ref)) {
            res.insert(narrowed);
          }
          auto constIt = constantSets.find(target);
          if (constIt != constantSets.end()) {
            res.insert(constIt->second.begin(), constIt->second.end());
          }
        }
      }
      if (followConditionalDependencies) {
        for (const auto &[source, targets] : conditionalDependencyEdges) {
          if ((!source.overlaps(*currRef) &&
               !(expandDescendants && hasOverlappingPrefix(source, *currRef))) ||
              !isCompatibleIndexedAlternative(source, ref)) {
            continue;
          }
          for (const SourceRef &target : targets) {
            if (!isCompatibleIndexedAlternative(target, ref)) {
              continue;
            }
            SourceRef narrowed = target.narrowRanges(*currRef);
            if (!visited.contains(narrowed)) {
              worklist.push_back(narrowed);
            }
            if (!narrowed.overlaps(ref)) {
              res.insert(narrowed);
            }
          }
        }
      }
      for (const auto &[constantSource, constants] : constantSets) {
        if ((constantSource.overlaps(*currRef) ||
             (expandDescendants && hasOverlappingPrefix(constantSource, *currRef))) &&
            isCompatibleIndexedAlternative(constantSource, ref)) {
          res.insert(constants.begin(), constants.end());
        }
      }
      // Constraints on a parent also constrain aggregate-valued and scalar descendants.
      currRef = currRef->getParentPrefix();
    }
  }

  DenseMap<SourceRef, SourceRef> preferredAliases;
  auto root = ref.getRoot();
  StructDefOp queriedStruct =
      succeeded(root) ? root->getParentRegion()->getParentOfType<StructDefOp>() : nullptr;
  auto rankAlias = [queriedStruct](const SourceRef &candidate) {
    if (candidate.isBlockArgument() && *candidate.getInputNum() > 0) {
      auto candidateRoot = candidate.getRoot();
      auto candidateStruct = succeeded(candidateRoot)
                                 ? candidateRoot->getParentRegion()->getParentOfType<StructDefOp>()
                                 : nullptr;
      return candidateStruct == queriedStruct ? 0 : 1;
    }
    if (candidate.isCreateStructOp()) {
      return 2;
    }
    if (candidate.isBlockArgument()) {
      return 3;
    }
    return 4;
  };
  for (const auto *leaderIt : directAliases) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    auto memberIt = directAliases.member_begin(*leaderIt);
    SourceRef preferred = *memberIt;
    for (; memberIt != directAliases.member_end(); ++memberIt) {
      if (rankAlias(*memberIt) < rankAlias(preferred) ||
          (rankAlias(*memberIt) == rankAlias(preferred) && *memberIt < preferred)) {
        preferred = *memberIt;
      }
    }
    for (memberIt = directAliases.member_begin(*leaderIt); memberIt != directAliases.member_end();
         ++memberIt) {
      preferredAliases.try_emplace(*memberIt, preferred);
    }
  }

  SourceRefSet normalized;
  if (ctx.runIntraproceduralAnalysis() &&
      llvm::any_of(res, [](const SourceRef &resultRef) { return resultRef.isNonDetOp(); })) {
    SourceRefSet equivalentInputs;
    for (const auto *candidate : signalSets) {
      const SourceRef &candidateRef = candidate->getData();
      if (candidateRef.isBlockArgument() && *candidateRef.getInputNum() > 0) {
        equivalentInputs.insert(candidateRef);
      }
    }
    res.insert(equivalentInputs.begin(), equivalentInputs.end());
  }

  for (const SourceRef &resultRef : res) {
    auto aliasIt = preferredAliases.find(resultRef);
    // Logical inputs remain distinct dependencies even when constrained equal to each other.
    if (aliasIt == preferredAliases.end() ||
        (resultRef.isBlockArgument() && *resultRef.getInputNum() > 0)) {
      normalized.insert(resultRef);
    } else if (!aliasIt->second.overlaps(ref)) {
      normalized.insert(aliasIt->second);
    } else if (!resultRef.overlaps(ref)) {
      normalized.insert(resultRef);
    }
  }

  const bool hasInputDependency = llvm::any_of(normalized, [](const SourceRef &resultRef) {
    return resultRef.isBlockArgument() && *resultRef.getInputNum() > 0;
  });
  if (hasInputDependency) {
    auto isLogicalComponentPath = [](const SourceRef &resultRef) {
      if (!resultRef.isBlockArgument() || *resultRef.getInputNum() != 0) {
        return false;
      }
      return llvm::any_of(resultRef.getPath(), [](const SourceRefIndex &index) {
        return index.isMember() && llvm::isa<StructType>(index.getMember().getType());
      });
    };
    const bool queryIsPodStorage = llvm::isa<PodType>(ref.getType());
    const bool hasLogicalComponentPath =
        !queryIsPodStorage && llvm::any_of(normalized, isLogicalComponentPath);
    SourceRefSet terminalDependencies;
    auto isTerminal = [&](const SourceRef &resultRef) {
      if (resultRef.isConstant() || (isLogicalComponentPath(resultRef) && !queryIsPodStorage)) {
        return true;
      }
      if (!resultRef.isBlockArgument() || *resultRef.getInputNum() == 0) {
        return false;
      }
      return !hasLogicalComponentPath || llvm::none_of(resultRef.getPath(), [](const auto &index) {
        return index.isIndexRange();
      });
    };
    for (const SourceRef &resultRef : normalized) {
      if (isTerminal(resultRef)) {
        terminalDependencies.insert(resultRef);
      }
    }
    normalized = std::move(terminalDependencies);
  }

  auto isStrictRangeSubset = [](const SourceRef &candidate, const SourceRef &other) {
    auto candidateRoot = candidate.getRoot();
    auto otherRoot = other.getRoot();
    if (failed(candidateRoot) || failed(otherRoot) || *candidateRoot != *otherRoot ||
        candidate.getPath().size() != other.getPath().size()) {
      return false;
    }

    bool isStrict = false;
    for (auto [candidateIndex, otherIndex] : llvm::zip(candidate.getPath(), other.getPath())) {
      if (candidateIndex == otherIndex) {
        continue;
      }
      if (!candidateIndex.isIndexRange() || !otherIndex.isIndexRange()) {
        return false;
      }
      auto [candidateLower, candidateUpper] = candidateIndex.getIndexRange();
      auto [otherLower, otherUpper] = otherIndex.getIndexRange();
      if (candidateLower < otherLower || candidateUpper > otherUpper) {
        return false;
      }
      isStrict = true;
    }
    return isStrict;
  };
  SourceRefSet minimalRanges;
  for (const SourceRef &candidate : normalized) {
    if (llvm::none_of(normalized, [&](const SourceRef &other) {
      return isStrictRangeSubset(candidate, other);
    })) {
      minimalRanges.insert(candidate);
    }
  }
  return minimalRanges;
}

//===----------------------------------------------------------------------===//
// ConstraintDependencyGraphModuleAnalysis
//===----------------------------------------------------------------------===//

LogicalResult ConstraintDependencyGraphModuleAnalysis::validateInput(ModuleOp module) const {
  if (failed(SourceRefAnalysis::validateInput(module))) {
    return failure();
  }
  auto result = module.walk([](Operation *op) -> WalkResult {
    if (op->getName().getDialectNamespace() == "cf") {
      return op->emitOpError(
          "constraint dependency graph analysis requires SCF control flow; "
          "operations from the 'cf' dialect are unsupported"
      );
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// ConstraintDependencyGraphStructAnalysis
//===----------------------------------------------------------------------===//

mlir::LogicalResult ConstraintDependencyGraphStructAnalysis::runAnalysis(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager,
    const CDGAnalysisContext &ctx
) {
  auto result = ConstraintDependencyGraph::compute(
      getModule(), getStruct(), solver, moduleAnalysisManager, ctx
  );
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  setResult(ctx, std::move(*result));
  return mlir::success();
}

} // namespace llzk
