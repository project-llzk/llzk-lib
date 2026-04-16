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
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/Hash.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Debug.h>

#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-cdg"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace constrain;
using namespace function;

static bool isOperationLive(DataFlowSolver &solver, Operation *op) {
  if (!op->getBlock()) {
    return true;
  }
  if (const auto *exec =
          solver.lookupState<mlir::dataflow::Executable>(solver.getProgramPointBefore(op->getBlock()
          ))) {
    return exec->isLive();
  }
  return true;
}

/* SourceRefAnalysis */

const SourceRefAnalysis::Lattice *SourceRefAnalysis::getLattice(DataFlowSolver &solver, Value val) {
  return solver.lookupState<Lattice>(val);
}

SourceRefLatticeValue
SourceRefAnalysis::getValueState(DataFlowSolver &solver, Operation * /*contextOp*/, Value val) {
  if (const auto *state = getLattice(solver, val)) {
    return state->getValue();
  }
  return SourceRefLattice::getDefaultValue(val);
}

mlir::FailureOr<SourceRefLatticeValue>
SourceRefAnalysis::getWriteTargetState(DataFlowSolver &solver, Operation *op) {
  llvm::SmallDenseMap<Value, SourceRefLatticeValue, 4> operandVals;
  for (Value operand : op->getOperands()) {
    operandVals[operand] = getValueState(solver, op, operand);
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

  if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    if (llvm::isa<WriteArrayOp, InsertArrayOp>(arrayAccessOp)) {
      auto array = arrayAccessOp.getArrRef();
      auto it = operandVals.find(array);
      ensure(it != operandVals.end(), "improperly constructed operandVals map");
      const auto &currVals = it->second;

      std::vector<SourceRefIndex> indices;
      for (unsigned i = 0; i < arrayAccessOp.getIndices().size(); ++i) {
        auto idxOperand = arrayAccessOp.getIndices()[i];
        auto idxIt = operandVals.find(idxOperand);
        ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
        const auto &idxVals = idxIt->second;

        if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstant()) {
          indices.emplace_back(*idxVals.getSingleValue().getConstantValue());
        } else {
          auto arrayType = llvm::dyn_cast<ArrayType>(array.getType());
          auto lower = APInt::getZero(64);
          APInt upper(64, arrayType.getDimSize(i));
          indices.emplace_back(lower, upper);
        }
      }

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
  if (auto value = llvm::dyn_cast_if_present<Value>(lattice->getAnchor())) {
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
    for (unsigned i = 0; i < elements.size(); i++) {
      (void)newArrayVal.getElemFlatIdx(i).setValue(operandVals.at(elements[i])->getValue());
    }
    propagateIfChanged(results.front(), results.front()->setValue(newArrayVal));
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
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    translation[SourceRef(funcOp.getArgument(i))] =
        static_cast<const Lattice *>(operandLattices[i])->getValue();
  }

  for (auto [result, resultLattice] : llvm::zip(call->getResults(), resultLattices)) {
    (void)result;
    SourceRefLatticeValue combined;
    unsigned resultNum = llvm::cast<OpResult>(result).getResultNumber();
    for (Operation *returnSite : returnSites) {
      auto retVal = static_cast<const Lattice *>(getLatticeElementFor(
                                                     getProgramPointAfter(call.getOperation()),
                                                     returnSite->getOperand(resultNum)
                                                 ))
                        ->getValue();
      auto [translatedVal, _] = retVal.translate(translation);
      (void)combined.update(translatedVal);
    }
    propagateIfChanged(resultLattice, static_cast<Lattice *>(resultLattice)->setValue(combined));
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

  std::vector<SourceRefIndex> indices;
  for (unsigned i = 0; i < arrayAccessOp.getIndices().size(); ++i) {
    auto idxOperand = arrayAccessOp.getIndices()[i];
    auto idxIt = operandVals.find(idxOperand);
    ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
    const auto &idxVals = idxIt->second->getValue();

    if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstant()) {
      indices.emplace_back(*idxVals.getSingleValue().getConstantValue());
    } else {
      auto arrayType = llvm::dyn_cast<ArrayType>(array.getType());
      auto lower = APInt::getZero(64);
      APInt upper(64, arrayType.getDimSize(i));
      indices.emplace_back(lower, upper);
    }
  }

  auto newValsRes = currVals.extract(indices);
  ensure(succeeded(newValsRes), "could not create SourceRef child for array access");
  auto [newVals, _] = *newValsRes;
  if (llvm::isa<ReadArrayOp, WriteArrayOp>(arrayAccessOp)) {
    ensure(newVals.isScalar(), "array read/write must produce a scalar value");
  }
  return newVals;
}

/* ConstraintDependencyGraph */

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
  for (auto it = signalSets.begin(); it != signalSets.end(); it++) {
    if (!it->isLeader()) {
      continue;
    }

    std::set<SourceRef> sortedMembers;
    for (auto mit = signalSets.member_begin(it); mit != signalSets.member_end(); mit++) {
      sortedMembers.insert(*mit);
    }

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

  for (auto it = sortedSets.begin(); it != sortedSets.end();) {
    os << "\n    { ";
    for (auto mit = it->begin(); mit != it->end();) {
      os << *mit;
      mit++;
      if (mit != it->end()) {
        os << ", ";
      }
    }

    it++;
    if (it == sortedSets.end()) {
      os << " }\n";
    } else {
      os << " },";
    }
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
    if (!isOperationLive(solver, op)) {
      return;
    }

    for (Value operand : op->getOperands()) {
      auto operandRefs = SourceRefAnalysis::getValueState(solver, op, operand).foldToScalar();
      for (const SourceRef &ref : operandRefs) {
        ref2Val[ref].insert(operand);
      }
    }
    for (Value result : op->getResults()) {
      auto resultRefs = SourceRefAnalysis::getValueState(solver, op, result).foldToScalar();
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
    if (!isOperationLive(solver, fnCall.getOperation())) {
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
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      SourceRef prefix(fn.getArgument(i));
      Value operand = fnCall.getOperand(i);
      SourceRefLatticeValue val =
          SourceRefAnalysis::getValueState(solver, fnCall.getOperation(), operand);
      translations.push_back({prefix, val});
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
    for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
      if (!lit->isLeader()) {
        continue;
      }
      auto leader = lit->getData();
      for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
        signalSets.unionSets(leader, *mit);
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

void ConstraintDependencyGraph::walkConstrainOp(
    mlir::DataFlowSolver &solver, mlir::Operation *emitOp
) {
  std::vector<SourceRef> signalUsages, constUsages;

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = SourceRefAnalysis::getValueState(solver, emitOp, operand);
    for (const auto &ref : latticeVal.foldToScalar()) {
      if (ref.isConstant()) {
        constUsages.push_back(ref);
      } else {
        signalUsages.push_back(ref);
      }
    }
  }

  // Compute a transitive closure over the signals.
  if (!signalUsages.empty()) {
    auto it = signalUsages.begin();
    auto leader = signalSets.getOrInsertLeaderValue(*it);
    for (it++; it != signalUsages.end(); it++) {
      signalSets.unionSets(leader, *it);
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

  for (auto leaderIt = signalSets.begin(); leaderIt != signalSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<SourceRef> translatedSignals, translatedConsts;
    for (auto mit = signalSets.member_begin(leaderIt); mit != signalSets.member_end(); mit++) {
      auto member = translate(*mit);
      if (mlir::failed(member)) {
        continue;
      }
      for (const auto &ref : *member) {
        if (ref.isConstant()) {
          translatedConsts.push_back(ref);
        } else {
          translatedSignals.push_back(ref);
        }
      }
      // Also add the constants from the original CDG
      if (auto it = constantSets.find(*mit); it != constantSets.end()) {
        const auto &origConstSet = it->second;
        translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
      }
    }

    if (translatedSignals.empty()) {
      continue;
    }

    // Now we can insert the translated signals
    auto it = translatedSignals.begin();
    auto leader = *it;
    res.signalSets.insert(leader);
    for (it++; it != translatedSignals.end(); it++) {
      res.signalSets.insert(*it);
      res.signalSets.unionSets(leader, *it);
    }

    // And update the constant references
    for (auto &ref : translatedSignals) {
      res.constantSets[ref].insert(translatedConsts.begin(), translatedConsts.end());
    }
  }

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
  auto currRef = mlir::FailureOr<SourceRef>(ref);
  while (mlir::succeeded(currRef)) {
    // Add signals
    for (auto it = signalSets.findLeader(*currRef); it != signalSets.member_end(); it++) {
      if (currRef.value() != *it) {
        res.insert(*it);
      }
    }
    // Add constants
    auto constIt = constantSets.find(*currRef);
    if (constIt != constantSets.end()) {
      res.insert(constIt->second.begin(), constIt->second.end());
    }
    // Go to parent
    currRef = currRef->getParentPrefix();
  }
  return res;
}

/* ConstraintDependencyGraphStructAnalysis */

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
