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
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Util/Hash.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Debug.h>

#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-cdg"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace cast;
using namespace component;
using namespace constrain;
using namespace function;
using namespace pod;

namespace {
bool isInMaybeSkippedScfRegion(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    if (llvm::isa<FuncDefOp>(parent)) {
      return false;
    }
    if (llvm::isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent)) {
      return true;
    }
  }
  return false;
}

std::optional<std::pair<APInt, APInt>> getStaticLoopIndexRange(Value index) {
  if (auto toIndex = index.getDefiningOp<FeltToIndexOp>()) {
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
      return std::pair(APInt(64, lower.value()), APInt(64, upper.value()));
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
      cmp.getLhs() != beforeArgument) {
    return std::nullopt;
  }
  auto upper = cmp.getRhs().getDefiningOp<felt::FeltConstantOp>();
  if (!upper) {
    return std::nullopt;
  }
  return std::pair(lower.getValueAPInt(), upper.getValueAPInt());
}

std::optional<Value> getFullyOverwrittenLoopArrayInitializer(Value value) {
  auto read = value.getDefiningOp<ReadArrayOp>();
  auto whileResult = read ? llvm::dyn_cast<OpResult>(read.getArrRef()) : OpResult();
  auto whileOp = whileResult ? whileResult.getDefiningOp<scf::WhileOp>() : scf::WhileOp();
  if (!whileOp || whileResult.getResultNumber() >= whileOp.getInits().size()) {
    return std::nullopt;
  }

  Value initializer = whileOp.getInits()[whileResult.getResultNumber()];
  auto arrayType = llvm::dyn_cast<ArrayType>(initializer.getType());
  if (!initializer.getDefiningOp<NonDetOp>() || !arrayType || !arrayType.hasStaticShape() ||
      arrayType.getShape().size() != 1) {
    return std::nullopt;
  }

  Value carried = whileOp.getAfter().front().getArgument(whileResult.getResultNumber());
  bool coversArray = false;
  whileOp.getAfter().walk([&](WriteArrayOp write) {
    if (write.getArrRef() != carried || write.getIndices().size() != 1) {
      return;
    }
    auto range = getStaticLoopIndexRange(write.getIndices().front());
    coversArray |= range && range->first.isZero() &&
                   range->second == APInt(range->second.getBitWidth(), arrayType.getDimSize(0));
  });
  return coversArray ? std::optional<Value>(initializer) : std::nullopt;
}

bool isZeroInitializerWrite(const SourceRef &ref, Value initializer) {
  auto constant = ref.getConstantValue();
  auto value = ref.getConstant();
  if (failed(constant) || *constant != 0 || failed(value)) {
    return false;
  }
  return llvm::any_of(value->getUsers(), [initializer](Operation *user) {
    auto write = llvm::dyn_cast<WriteArrayOp>(user);
    return write && write.getArrRef() == initializer;
  });
}

llvm::SmallVector<std::pair<uint64_t, uint64_t>>
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

bool isCompatibleIndexedAlternative(const SourceRef &candidate, const SourceRef &selection) {
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

SourceRefLatticeValue createShapedArrayValue(Value rootValue, ArrayType arrayTy) {
  SourceRefLatticeValue result(arrayTy.getShape());
  ArrayIndexGen indexGen = ArrayIndexGen::from(arrayTy);
  SourceRef root = *SourceRefLattice::getSourceRef(rootValue);
  for (size_t i = 0; i < result.getArraySize(); ++i) {
    auto indices = indexGen.delinearize(i, rootValue.getContext());
    ensure(indices.has_value(), "could not delinearize aggregate array element index");
    SourceRef element = root;
    for (Attribute attr : *indices) {
      auto child = element.createChild(SourceRefIndex(llvm::cast<IntegerAttr>(attr).getValue()));
      ensure(succeeded(child), "could not create aggregate array element SourceRef");
      element = *child;
    }
    (void)result.getElemFlatIdx(i).setValue(SourceRefLatticeValue(element));
  }
  return result;
}

std::optional<SourceRef> getNeutralWhileInitializer(Value value) {
  auto result = llvm::dyn_cast<OpResult>(value);
  auto whileOp =
      result ? llvm::dyn_cast_if_present<scf::WhileOp>(result.getDefiningOp()) : scf::WhileOp();
  if (!whileOp) {
    return std::nullopt;
  }

  unsigned resultNumber = result.getResultNumber();
  Value init = whileOp.getInits()[resultNumber];
  auto zero = init.getDefiningOp<felt::FeltConstantOp>();
  if (!zero || !zero.getValueAPInt().isZero()) {
    return std::nullopt;
  }

  Value carried = whileOp.getAfter().front().getArgument(resultNumber);
  Value yielded = whileOp.getYieldOp().getOperand(resultNumber);
  auto add = yielded.getDefiningOp<felt::AddFeltOp>();
  if (!add || !llvm::is_contained(add->getOperands(), carried)) {
    return std::nullopt;
  }
  return SourceRef(zero);
}

} // namespace

/* SourceRefAnalysis */

/// Solver-owned state for aggregate storage dependencies discovered by `SourceRefAnalysis`.
///
/// The ordinary `SourceRefLattice` records the storage addresses represented by each SSA value.
/// This state separately records values written to those addresses and aliases introduced when
/// aggregates are assigned to other storage. It is anchored before the top-level operation so
/// static dependency queries can retrieve it from the solver without retaining a pointer to the
/// analysis instance. The state is created lazily when the analysis first records a storage write.
class SourceRefAnalysis::StorageState : public AnalysisState {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StorageState)

  using AnalysisState::AnalysisState;

  /// Associate this state with the top-level operation whose writes it models.
  void setTop(Operation *op) {
    ensure(top == nullptr || top == op, "storage state cannot span top-level operations");
    top = op;
  }

  /// Resolve known storage writes transitively, preserving unwritten and cyclic addresses.
  SourceRefLatticeValue
  resolveDependencies(const SourceRefLatticeValue &addresses, Operation *before) const;

  /// Materialize compact nondeterministic aggregates from their storage dependencies.
  SourceRefLatticeValue
  resolveValueDependencies(mlir::Value value, const SourceRefLatticeValue &fallback) const;

  /// Materialize compact nondeterministic aggregates from their final storage dependencies.
  SourceRefLatticeValue
  resolveValueDependencies(mlir::Value value, const SourceRefLatticeValue &fallback) const;

  /// Record a storage write for later dependency queries.
  void recordStorageWrite(
      Operation *op, size_t writeIndex, const SourceRefLatticeValue &addresses,
      const SourceRefLatticeValue &value, bool mayBeSkipped = false
  );

  /// Rebase an allocation/call result to the aggregate storage receiving it.
  void recordAggregateAlias(
      Operation *op, size_t aliasIndex, const SourceRefLatticeValue &source,
      const SourceRefLatticeValue &target, bool mayBeSkipped = false
  );

  void print(raw_ostream &os) const override { os << "SourceRefAnalysis::StorageState"; }

private:
  struct StorageWrite {
    SourceRefLatticeValue addresses;
    SourceRefLatticeValue value;
    bool mayBeSkipped;
  };

  struct AggregateAlias {
    SourceRefLatticeValue source;
    SourceRefLatticeValue target;
    bool mayBeSkipped;
  };

  /// Apply all known aggregate-storage aliases to a lattice value.
  static SourceRefLatticeValue
  canonicalize(const SourceRefLatticeValue &value, const TranslationMap &aliases);

  /// Materialize aliases established before `before` in program order.
  TranslationMap materializeAggregateAliases(Operation *before) const;

  /// Materialize storage contents at `before` by replaying earlier writes in IR order.
  llvm::DenseMap<SourceRef, SourceRefLatticeValue>
  materializeStoredValues(Operation *before, const TranslationMap &aliases) const;

  Operation *top = nullptr;
  llvm::DenseMap<Operation *, llvm::SmallVector<StorageWrite>> storageWrites;
  llvm::DenseMap<Operation *, llvm::SmallVector<AggregateAlias>> aggregateAliases;
};

const SourceRefAnalysis::Lattice *SourceRefAnalysis::getLattice(DataFlowSolver &solver, Value val) {
  return solver.lookupState<Lattice>(val);
}

SourceRefLatticeValue SourceRefAnalysis::getValueState(DataFlowSolver &solver, Value val) {
  if (const auto *state = getLattice(solver, val)) {
    return state->getValue();
  }
  return SourceRefLattice::getDefaultValue(val);
}

SourceRefLatticeValue SourceRefAnalysis::getDependencyState(DataFlowSolver &solver, Value val) {
  SourceRefLatticeValue result = getValueState(solver, val);
  Operation *top = val.getParentRegion()->getParentOp();
  while (top->getParentOp() != nullptr) {
    top = top->getParentOp();
  }
  if (const auto *state = solver.lookupState<StorageState>(solver.getProgramPointBefore(top))) {
    result = state->resolveValueDependencies(val, result);
  }
  if (std::optional<SourceRef> neutralInitializer = getNeutralWhileInitializer(val)) {
    (void)result.remove(*neutralInitializer);
    auto whileResult = llvm::cast<OpResult>(val);
    auto whileOp = llvm::cast<scf::WhileOp>(whileResult.getDefiningOp());
    Value carried = whileOp.getAfter().front().getArgument(whileResult.getResultNumber());
    auto add = whileOp.getYieldOp()
                   .getOperand(whileResult.getResultNumber())
                   .getDefiningOp<felt::AddFeltOp>();
    for (Value operand : add->getOperands()) {
      if (operand != carried) {
        (void)result.update(getDependencyState(solver, operand));
      }
    }
  }
  if (std::optional<Value> initializer = getFullyOverwrittenLoopArrayInitializer(val)) {
    for (const SourceRef &ref : result.foldToScalar()) {
      if (isZeroInitializerWrite(ref, *initializer)) {
        (void)result.remove(ref);
      }
    }
  }
  return result;
}

SourceRefAnalysis::StorageState *SourceRefAnalysis::getStorageState(Operation *op) {
  while (op->getParentOp() != nullptr) {
    op = op->getParentOp();
  }
  auto *state = getOrCreate<StorageState>(getProgramPointBefore(op));
  state->setTop(op);
  return state;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveValueDependencies(
    Value value, const SourceRefLatticeValue &fallback
) const {
  auto result = llvm::dyn_cast<OpResult>(value);
  auto nondet = result ? result.getDefiningOp<NonDetOp>() : NonDetOp();
  auto arrayTy = nondet ? llvm::dyn_cast<ArrayType>(value.getType()) : ArrayType();
  Operation *before = nullptr;
  if (Operation *defOp = value.getDefiningOp()) {
    auto memberAccess = llvm::dyn_cast<MemberRefOpInterface>(defOp);
    auto podAccess = llvm::dyn_cast<PodAccessOpInterface>(defOp);
    if ((memberAccess && memberAccess.isRead()) || (podAccess && podAccess.isRead()) ||
        llvm::isa<ReadArrayOp, ExtractArrayOp>(defOp)) {
      before = defOp;
    }
  }
  constexpr size_t maxPreciselyResolvedElements = 64;
  if (!arrayTy || !arrayTy.hasStaticShape() ||
      std::cmp_greater(arrayTy.getNumElements(), maxPreciselyResolvedElements)) {
    return resolveDependencies(fallback, before);
  }
  // Aggregate allocations model storage populated by following operations, so their dependencies
  // intentionally use the complete write history. Storage reads use their defining operation as
  // the cutoff above.
  SourceRefLatticeValue resolved =
      resolveDependencies(createShapedArrayValue(value, arrayTy), before);
  for (size_t i = 0; i < resolved.getArraySize(); ++i) {
    SourceRefLatticeValue &element = resolved.getElemFlatIdx(i);
    SourceRefSet refs = element.foldToScalar();
    if (llvm::any_of(refs, [](const SourceRef &ref) { return !ref.isConstant(); })) {
      for (const SourceRef &ref : refs) {
        auto constant = ref.getConstantValue();
        if (succeeded(constant) && *constant == 0) {
          (void)element.remove(ref);
        }
      }
    }
  }
  return resolved;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::canonicalize(
    const SourceRefLatticeValue &value, const TranslationMap &aliases
) {
  SourceRefLatticeValue canonical = value;
  for (size_t i = 0; i < aliases.size(); ++i) {
    auto [next, changed] = canonical.replacePrefixes(aliases);
    canonical = std::move(next);
    if (changed == ChangeResult::NoChange) {
      break;
    }
  }
  return canonical;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveDependencies(
    const SourceRefLatticeValue &addresses, Operation *before
) const {
  const TranslationMap aliases = materializeAggregateAliases(before);
  const llvm::DenseMap<SourceRef, SourceRefLatticeValue> storedValues =
      materializeStoredValues(before, aliases);
  std::function<
      SourceRefLatticeValue(const SourceRefLatticeValue &, const SourceRef &, const SourceRef &)>
      projectChild = [&](const SourceRefLatticeValue &value, const SourceRef &storedAddress,
                         const SourceRef &readAddress) {
    if (!readAddress.isValidPrefix(storedAddress)) {
      return value;
    }
    if (value.isArray()) {
      SourceRefLatticeValue result(value.getArrayShape());
      for (size_t i = 0; i < value.getArraySize(); ++i) {
        (void)result.getElemFlatIdx(i).setValue(
            projectChild(value.getElemFlatIdx(i), storedAddress, readAddress)
        );
      }
      return result;
    }

    SourceRefLatticeValue result;
    for (const SourceRef &ref : value.getScalarValue()) {
      if (auto translated = readAddress.translate(storedAddress, ref); succeeded(translated)) {
        (void)result.insert(*translated);
      } else {
        (void)result.insert(ref);
      }
    }
    return result;
  };
  llvm::DenseSet<SourceRef> active;
  std::function<SourceRefLatticeValue(const SourceRefLatticeValue &)> resolve =
      [&](const SourceRefLatticeValue &input) {
    SourceRefLatticeValue addressValue = canonicalize(input, aliases);
    if (addressValue.isArray()) {
      SourceRefLatticeValue result(addressValue.getArrayShape());
      for (size_t i = 0; i < addressValue.getArraySize(); ++i) {
        (void)result.getElemFlatIdx(i).setValue(resolve(addressValue.getElemFlatIdx(i)));
      }
      return result;
    }

    SourceRefLatticeValue result;
    for (const SourceRef &address : addressValue.getScalarValue()) {
      if (!active.insert(address).second) {
        if (addressValue.isSingleValue()) {
          (void)result.insert(address);
        }
        continue;
      }

      bool foundWrite = false;
      SourceRefLatticeValue writtenValues;
      for (const auto &[storedAddress, storedValue] : storedValues) {
        auto storedRoot = storedAddress.getRoot();
        auto addressRoot = address.getRoot();
        if (failed(storedRoot) || failed(addressRoot) || *storedRoot != *addressRoot ||
            (!storedAddress.overlaps(address) && !address.isValidPrefix(storedAddress))) {
          continue;
        }
        foundWrite = true;
        (void)writtenValues.update(projectChild(storedValue, storedAddress, address));
      }
      if (foundWrite) {
        SourceRefLatticeValue resolvedValues = resolve(writtenValues);
        (void)result.update(resolvedValues);
      }
      if (!foundWrite) {
        (void)result.insert(address);
      }
      active.erase(address);
    }
    return result;
  };

  return resolve(addresses);
}

void SourceRefAnalysis::StorageState::recordStorageWrite(
    Operation *op, size_t writeIndex, const SourceRefLatticeValue &addresses,
    const SourceRefLatticeValue &value, bool mayBeSkipped
) {
  auto &writes = storageWrites[op];
  StorageWrite write {addresses, value, mayBeSkipped};
  if (writeIndex == writes.size()) {
    writes.push_back(std::move(write));
    return;
  }
  ensure(writeIndex < writes.size(), "storage writes must be recorded in stable operation order");
  writes[writeIndex] = std::move(write);
}

llvm::DenseMap<SourceRef, SourceRefLatticeValue>
SourceRefAnalysis::StorageState::materializeStoredValues(
    Operation *before, const TranslationMap &aliases
) const {
  llvm::DenseMap<SourceRef, SourceRefLatticeValue> storedValues;
  ensure(top != nullptr, "storage state must be associated with a top-level operation");

  auto getArrayElementAddress = [](const SourceRef &root, size_t flatIndex) {
    auto arrayType = llvm::dyn_cast<ArrayType>(root.getType());
    ensure(arrayType && arrayType.hasStaticShape(), "shaped storage write requires static array");
    ArrayIndexGen indexGen = ArrayIndexGen::from(arrayType);
    auto indices =
        indexGen.delinearize(checkedCast<int64_t>(flatIndex), root.getType().getContext());
    ensure(indices.has_value(), "could not delinearize shaped storage write");
    SourceRef address = root;
    for (Attribute attr : *indices) {
      auto child = address.createChild(SourceRefIndex(llvm::cast<IntegerAttr>(attr).getValue()));
      ensure(succeeded(child), "could not create shaped storage address");
      address = *child;
    }
    return address;
  };

  auto applyWrite = [&](const SourceRef &address, const SourceRefLatticeValue &value,
                        bool maySkip) {
    auto [it, inserted] = storedValues.try_emplace(address, value);
    if (!inserted) {
      if (maySkip) {
        (void)it->second.update(value);
      } else {
        (void)it->second.setValue(value);
      }
    }
  };

  std::function<void(const SourceRef &, const SourceRefLatticeValue &, bool)> materializeWrite =
      [&](const SourceRef &address, const SourceRefLatticeValue &value, bool maySkip) {
    SourceRefLatticeValue canonicalValue = canonicalize(value, aliases);
    auto arrayType = llvm::dyn_cast<ArrayType>(address.getType());
    if (canonicalValue.isArray() && arrayType && arrayType.hasStaticShape() &&
        std::cmp_equal(canonicalValue.getArraySize(), arrayType.getNumElements())) {
      for (size_t i = 0; i < canonicalValue.getArraySize(); ++i) {
        materializeWrite(
            getArrayElementAddress(address, i), canonicalValue.getElemFlatIdx(i), maySkip
        );
      }
      return;
    }
    if (canonicalValue.isScalar() && canonicalValue.getScalarValue().contains(address)) {
      const bool hasPriorContents = llvm::any_of(storedValues, [&](const auto &entry) {
        return entry.first == address || entry.first.isValidPrefix(address);
      });
      if (hasPriorContents || canonicalValue.isSingleValue()) {
        (void)canonicalValue.getScalarValue().erase(address);
        if (canonicalValue.getScalarValue().empty()) {
          return;
        }
      }
    }
    applyWrite(address, canonicalValue, maySkip);
  };

  (void)top->walk([&](Operation *op) {
    if (op == before) {
      return WalkResult::interrupt();
    }
    auto writes = storageWrites.find(op);
    if (writes == storageWrites.end()) {
      return WalkResult::advance();
    }
    for (const StorageWrite &write : writes->second) {
      for (const SourceRef &address : canonicalize(write.addresses, aliases).foldToScalar()) {
        materializeWrite(address, write.value, write.mayBeSkipped);
      }
    }
    return WalkResult::advance();
  });
  return storedValues;
}

void SourceRefAnalysis::StorageState::recordAggregateAlias(
    Operation *op, size_t aliasIndex, const SourceRefLatticeValue &source,
    const SourceRefLatticeValue &target, bool mayBeSkipped
) {
  auto &aliases = aggregateAliases[op];
  AggregateAlias alias {source, target, mayBeSkipped};
  if (aliasIndex == aliases.size()) {
    aliases.push_back(std::move(alias));
    return;
  }
  ensure(aliasIndex < aliases.size(), "aggregate aliases must use stable operation order");
  aliases[aliasIndex] = std::move(alias);
}

TranslationMap
SourceRefAnalysis::StorageState::materializeAggregateAliases(Operation *before) const {
  TranslationMap aliases;
  ensure(top != nullptr, "storage state must be associated with a top-level operation");

  auto getArrayElementTarget = [](const SourceRef &root, size_t flatIndex) {
    auto arrayType = llvm::dyn_cast<ArrayType>(root.getType());
    ensure(arrayType && arrayType.hasStaticShape(), "array alias target requires static shape");
    ArrayIndexGen indexGen = ArrayIndexGen::from(arrayType);
    auto indices =
        indexGen.delinearize(checkedCast<int64_t>(flatIndex), root.getType().getContext());
    ensure(indices.has_value(), "could not delinearize aggregate alias target");
    SourceRef target = root;
    for (Attribute attr : *indices) {
      auto child = target.createChild(SourceRefIndex(llvm::cast<IntegerAttr>(attr).getValue()));
      ensure(succeeded(child), "could not create aggregate alias target child");
      target = *child;
    }
    return target;
  };

  std::function<void(const SourceRefLatticeValue &, const SourceRefLatticeValue &, bool)> addAlias =
      [&](const SourceRefLatticeValue &source, const SourceRefLatticeValue &target,
          bool mayBeSkipped) {
    SourceRefLatticeValue canonicalSource = canonicalize(source, aliases);
    SourceRefLatticeValue canonicalTarget = canonicalize(target, aliases);
    if (canonicalSource.isArray() && canonicalTarget.isSingleValue()) {
      const SourceRef &targetRoot = canonicalTarget.getSingleValue();
      auto targetType = llvm::dyn_cast<ArrayType>(targetRoot.getType());
      if (!targetType || !targetType.hasStaticShape() ||
          !std::cmp_equal(canonicalSource.getArraySize(), targetType.getNumElements())) {
        return;
      }
      for (size_t i = 0; i < canonicalSource.getArraySize(); ++i) {
        addAlias(
            canonicalSource.getElemFlatIdx(i),
            SourceRefLatticeValue(getArrayElementTarget(targetRoot, i)), mayBeSkipped
        );
      }
      return;
    }
    if (!canonicalSource.isScalar() || !canonicalTarget.isSingleValue()) {
      return;
    }

    const SourceRef &targetRef = canonicalTarget.getSingleValue();
    for (const SourceRef &sourceRef : canonicalSource.getScalarValue()) {
      if (sourceRef == targetRef || !sourceRef.isRooted() ||
          !llvm::isa<ArrayType, StructType, PodType>(sourceRef.getType())) {
        continue;
      }
      auto sourceRoot = sourceRef.getRoot();
      if (failed(sourceRoot)) {
        continue;
      }
      Operation *defOp = sourceRoot->getDefiningOp();
      if (defOp == nullptr || !llvm::isa<CallOp, NewPodOp, CreateArrayOp, NonDetOp>(defOp)) {
        continue;
      }
      SourceRefLatticeValue aliasTargets = canonicalTarget;
      if (mayBeSkipped) {
        (void)aliasTargets.insert(sourceRef);
      }
      aliases[sourceRef] = std::move(aliasTargets);
    }
  };

  (void)top->walk([&](Operation *op) {
    if (op == before) {
      return WalkResult::interrupt();
    }
    auto events = aggregateAliases.find(op);
    if (events != aggregateAliases.end()) {
      for (const AggregateAlias &event : events->second) {
        addAlias(event.source, event.target, event.mayBeSkipped);
      }
    }
    return WalkResult::advance();
  });
  return aliases;
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

      std::vector<SourceRefIndex> indices;
      for (size_t i = 0; i < arrayAccessOp.getIndices().size(); ++i) {
        auto idxOperand = arrayAccessOp.getIndices()[i];
        auto idxIt = operandVals.find(idxOperand);
        ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
        const auto &idxVals = idxIt->second;

        if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstant()) {
          indices.emplace_back(*idxVals.getSingleValue().getConstantValue());
        } else if (auto bounds = getStaticLoopIndexRange(idxOperand)) {
          indices.emplace_back(bounds->first, bounds->second);
        } else {
          auto arrayType = llvm::dyn_cast<ArrayType>(array.getType());
          auto lower = APInt::getZero(64);
          assert(i <= std::numeric_limits<unsigned>::max() && "index too large");
          APInt upper(64, arrayType.getDimSize(static_cast<unsigned>(i)));
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
    if (auto arg = llvm::dyn_cast<BlockArgument>(value)) {
      Operation *parent = arg.getOwner()->getParentOp();
      if (parent && !llvm::isa<FunctionOpInterface>(parent) &&
          llvm::isa<RegionBranchOpInterface>(parent) &&
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
    } else {
      auto [memberVals, _] = *memberValsRes;
      auto writeOp = llvm::cast<MemberWriteOp>(op);
      SourceRefLatticeValue writeValue = operandVals.at(writeOp.getVal())->getValue();
      if (llvm::isa<ArrayType, StructType, PodType>(writeOp.getVal().getType())) {
        const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
        getStorageState(op)->recordStorageWrite(
            op, /*writeIndex=*/0, memberVals, writeValue, mayBeSkipped
        );
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, memberVals, mayBeSkipped
        );
      }
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
    } else {
      auto [podVals, _] = *podValsRes;
      auto writeOp = llvm::cast<WritePodOp>(op);
      SourceRefLatticeValue writeValue = operandVals.at(writeOp.getValue())->getValue();
      const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
      getStorageState(op)->recordStorageWrite(
          op, /*writeIndex=*/0, podVals, writeValue, mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(writeOp.getValue().getType())) {
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, podVals, mayBeSkipped
        );
      }
    }
    return success();
  }

  if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    if (llvm::isa<WriteArrayOp, InsertArrayOp>(op)) {
      auto *arrayLattice = getLatticeElement(arrayAccessOp.getArrRef());
      SourceRefLatticeValue updatedArray = arrayLattice->getValue();
      Value rvalue = op->getOperands().back();
      SourceRefLatticeValue writeValue = operandVals.at(rvalue)->getValue();
      const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
      std::vector<SourceRefIndex> indices;
      SourceRefLatticeValue writeTargets =
          arraySubdivisionOpUpdate(arrayAccessOp, operandVals, &indices);
      if (updatedArray.isArray()) {
        ChangeResult changed = updatedArray.write(indices, writeValue, mayBeSkipped);
        if (changed == ChangeResult::Change) {
          propagateIfChanged(arrayLattice, arrayLattice->setValue(updatedArray));
        }
      }
      getStorageState(op)->recordStorageWrite(
          op, /*writeIndex=*/0, writeTargets, writeValue, mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(rvalue.getType())) {
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, writeTargets, mayBeSkipped
        );
      }
    } else if (!results.empty()) {
      auto newVals = arraySubdivisionOpUpdate(arrayAccessOp, operandVals);
      propagateIfChanged(results.front(), results.front()->setValue(newVals));
    }
    return success();
  }

  if (auto createArray = llvm::dyn_cast<CreateArrayOp>(op)) {
    auto createArrayRes = createArray.getResult();
    const auto &elements = createArray.getElements();
    if (elements.empty()) {
      ArrayType arrayTy = createArray.getType();
      if (arrayTy.hasStaticShape()) {
        propagateIfChanged(
            results.front(),
            results.front()->setValue(createShapedArrayValue(createArrayRes, arrayTy))
        );
      } else {
        propagateIfChanged(
            results.front(),
            results.front()->setValue(SourceRef(llvm::cast<OpResult>(createArrayRes)))
        );
      }
      return success();
    }

    SourceRefLatticeValue newArrayVal(createArray.getType().getShape());
    for (size_t i = 0; i < elements.size(); i++) {
      (void)newArrayVal.getElemFlatIdx(i).setValue(operandVals.at(elements[i])->getValue());
    }
    propagateIfChanged(results.front(), results.front()->setValue(newArrayVal));
    return success();
  }

  if (auto newPod = llvm::dyn_cast<NewPodOp>(op)) {
    auto newPodValue = SourceRefLattice::getDefaultValue(newPod.getResult());
    propagateIfChanged(results.front(), results.front()->setValue(newPodValue));
    SourceRef podRoot(llvm::cast<OpResult>(newPod.getResult()));
    for (auto [idx, record] : llvm::enumerate(newPod.getInitializedRecordValues())) {
      auto recordRef =
          podRoot.createChild(SourceRefIndex(StringAttr::get(op->getContext(), record.name)));
      ensure(succeeded(recordRef), "could not create initialized POD record SourceRef");
      SourceRefLatticeValue recordValue = operands[idx]->getValue();
      StorageState *state = getStorageState(op);
      SourceRefLatticeValue recordAddress(*recordRef);
      const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
      state->recordStorageWrite(op, idx, recordAddress, recordValue, mayBeSkipped);
      if (llvm::isa<ArrayType, StructType, PodType>(record.value.getType())) {
        state->recordAggregateAlias(op, idx, recordValue, recordAddress, mayBeSkipped);
      }
    }
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
      SourceRefLatticeValue resultValue(*resultRef);
      propagateIfChanged(lattice, lattice->setValue(resultValue));
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
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    SourceRefLatticeValue argumentValue =
        static_cast<const Lattice *>(operandLattices[i])->getValue();
    if (!llvm::isa<ArrayType, StructType, PodType>(call->getOperand(i).getType())) {
      argumentValue = getStorageState(call.getOperation())
                          ->resolveDependencies(argumentValue, call.getOperation());
    }
    translation[SourceRef(funcOp.getArgument(i))] = argumentValue;
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
      SourceRefLatticeValue returnDependencies = retVal;
      if (!llvm::isa<ArrayType, StructType, PodType>(returnSite->getOperand(resultNum).getType())) {
        returnDependencies = getStorageState(returnSite)->resolveDependencies(retVal, returnSite);
      }
      auto [translatedVal, _] = returnDependencies.translate(translation);
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
    ArrayAccessOpInterface arrayAccessOp, const OperandValues &operandVals,
    std::vector<SourceRefIndex> *resolvedIndices
) {
  auto array = arrayAccessOp.getArrRef();
  auto it = operandVals.find(array);
  ensure(it != operandVals.end(), "improperly constructed operandVals map");
  const auto &currVals = it->second->getValue();

  std::vector<SourceRefIndex> indices;
  for (size_t i = 0; i < arrayAccessOp.getIndices().size(); ++i) {
    auto idxOperand = arrayAccessOp.getIndices()[i];
    auto idxIt = operandVals.find(idxOperand);
    ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
    const auto &idxVals = idxIt->second->getValue();

    if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstant()) {
      indices.emplace_back(*idxVals.getSingleValue().getConstantValue());
    } else if (auto bounds = getStaticLoopIndexRange(idxOperand)) {
      indices.emplace_back(bounds->first, bounds->second);
    } else {
      auto arrayType = llvm::dyn_cast<ArrayType>(array.getType());
      auto lower = APInt::getZero(64);
      assert(i <= std::numeric_limits<unsigned>::max() && "index too large");
      APInt upper(64, arrayType.getDimSize(static_cast<unsigned>(i)));
      indices.emplace_back(lower, upper);
    }
  }

  if (resolvedIndices != nullptr) {
    *resolvedIndices = indices;
    // Write-like operations only need the selected indices before updating the base lattice. Do
    // not try to extract the old value: uninitialized or partially shaped aggregate storage may
    // not yet support that read, while the subsequent write can still initialize it.
    return SourceRefLatticeValue();
  }
  auto newValsRes = currVals.extract(indices);
  if (failed(newValsRes)) {
    // Aggregate storage may conservatively fold to a scalar dependency set after control-flow or
    // alias joins. In that case an element read depends on the whole folded value; retaining those
    // dependencies is safer than manufacturing an ill-typed child path (and avoids a fatal error).
    return currVals;
  }
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
    if (!dataflow::isOperationLive(solver, op)) {
      return;
    }

    for (Value operand : op->getOperands()) {
      for (const SourceRefLatticeValue &state :
           {SourceRefAnalysis::getValueState(solver, operand),
            SourceRefAnalysis::getDependencyState(solver, operand)}) {
        for (const SourceRef &ref : state.foldToScalar()) {
          ref2Val[ref].insert(operand);
        }
      }
    }
    for (Value result : op->getResults()) {
      for (const SourceRefLatticeValue &state :
           {SourceRefAnalysis::getValueState(solver, result),
            SourceRefAnalysis::getDependencyState(solver, result)}) {
        for (const SourceRef &ref : state.foldToScalar()) {
          ref2Val[ref].insert(result);
        }
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
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      SourceRef prefix(fn.getArgument(i));
      Value operand = fnCall.getOperand(i);
      SourceRefLatticeValue val = llvm::isa<ArrayType, StructType, PodType>(operand.getType())
                                      ? SourceRefAnalysis::getValueState(solver, operand)
                                      : SourceRefAnalysis::getDependencyState(solver, operand);
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
    for (const auto &[source, targets] : translatedCDG.dependencyEdges) {
      dependencyEdges[source].insert(targets.begin(), targets.end());
    }
    auto &translatedAliases = translatedCDG.directAliases;
    for (auto leaderIt = translatedAliases.begin(); leaderIt != translatedAliases.end();
         ++leaderIt) {
      if (!leaderIt->isLeader()) {
        continue;
      }
      const SourceRef &leader = leaderIt->getData();
      for (auto memberIt = translatedAliases.member_begin(leaderIt);
           memberIt != translatedAliases.member_end(); ++memberIt) {
        directAliases.unionSets(leader, *memberIt);
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
  llvm::SmallVector<SourceRefSet, 2> operandSignals;

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = SourceRefAnalysis::getDependencyState(solver, operand);
    std::optional<SourceRef> neutralInitializer = getNeutralWhileInitializer(operand);
    SourceRefSet currentOperandSignals;
    for (const auto &ref : latticeVal.foldToScalar()) {
      if (ref.isConstant() || ref.isTemplateConstant()) {
        if (!neutralInitializer || ref != *neutralInitializer) {
          constUsages.push_back(ref);
        }
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
    for (const SourceRef &lhs : signalUsages) {
      for (const SourceRef &rhs : signalUsages) {
        if (lhs != rhs) {
          dependencyEdges[lhs].insert(rhs);
        }
      }
    }
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

  for (const auto &[source, targets] : dependencyEdges) {
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
            res.dependencyEdges[translatedSource].insert(translatedTarget);
          }
        }
      }
    }
  }

  for (auto leaderIt = directAliases.begin(); leaderIt != directAliases.end(); ++leaderIt) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    std::vector<SourceRef> translatedAliases;
    for (auto memberIt = directAliases.member_begin(leaderIt);
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
  llvm::SmallVector<SourceRef> worklist = {ref};

  // Overlapping aggregate ranges can bridge otherwise distinct equivalence sets. Follow those
  // bridges transitively so that, for example, `out[2] == child.out[2]` and a child constraint on
  // `child.out[0:N]` connect `out[2]` to the child's inputs.
  while (!worklist.empty()) {
    auto currRef = mlir::FailureOr<SourceRef>(worklist.pop_back_val());
    while (mlir::succeeded(currRef)) {
      if (!visited.insert(*currRef).second) {
        break;
      }

      for (const auto &[source, targets] : dependencyEdges) {
        if (!source.overlaps(*currRef) || !isCompatibleIndexedAlternative(source, ref)) {
          continue;
        }
        if (auto constIt = constantSets.find(source); constIt != constantSets.end()) {
          res.insert(constIt->second.begin(), constIt->second.end());
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
          auto constIt = constantSets.find(target);
          if (constIt != constantSets.end()) {
            res.insert(constIt->second.begin(), constIt->second.end());
          }
        }
      }
      for (const auto &[constantSource, constants] : constantSets) {
        if (constantSource.overlaps(*currRef) &&
            isCompatibleIndexedAlternative(constantSource, ref)) {
          res.insert(constants.begin(), constants.end());
        }
      }
      // Constraints on an aggregate also constrain its scalar descendants.
      currRef = currRef->getParentPrefix();
    }
  }

  DenseMap<SourceRef, SourceRef> preferredAliases;
  auto rankAlias = [](const SourceRef &candidate) {
    if (candidate.isBlockArgument() && *candidate.getInputNum() > 0) {
      return 0;
    }
    if (candidate.isCreateStructOp()) {
      return 1;
    }
    if (candidate.isBlockArgument()) {
      return 2;
    }
    return 3;
  };
  for (auto leaderIt = directAliases.begin(); leaderIt != directAliases.end(); ++leaderIt) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    auto memberIt = directAliases.member_begin(leaderIt);
    SourceRef preferred = *memberIt;
    for (; memberIt != directAliases.member_end(); ++memberIt) {
      if (rankAlias(*memberIt) < rankAlias(preferred) ||
          (rankAlias(*memberIt) == rankAlias(preferred) && *memberIt < preferred)) {
        preferred = *memberIt;
      }
    }
    for (memberIt = directAliases.member_begin(leaderIt); memberIt != directAliases.member_end();
         ++memberIt) {
      preferredAliases.try_emplace(*memberIt, preferred);
    }
  }

  SourceRefSet normalized;
  if (ctx.runIntraproceduralAnalysis() &&
      llvm::any_of(res, [](const SourceRef &resultRef) { return resultRef.isNonDetOp(); })) {
    SourceRefSet equivalentInputs;
    for (auto candidate = signalSets.begin(); candidate != signalSets.end(); ++candidate) {
      const SourceRef &candidateRef = candidate->getData();
      if (candidateRef.isBlockArgument() && *candidateRef.getInputNum() > 0) {
        equivalentInputs.insert(candidateRef);
      }
    }
    res.insert(equivalentInputs.begin(), equivalentInputs.end());
  }

  for (const SourceRef &resultRef : res) {
    auto aliasIt = preferredAliases.find(resultRef);
    if (aliasIt == preferredAliases.end()) {
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
    SourceRefSet terminalDependencies;
    llvm::copy_if(
        normalized, std::inserter(terminalDependencies, terminalDependencies.end()),
        [&](const SourceRef &resultRef) {
      return resultRef.isConstant() ||
             (resultRef.isBlockArgument() && *resultRef.getInputNum() > 0) ||
             isLogicalComponentPath(resultRef);
    }
    );
    normalized = std::move(terminalDependencies);
  }

  return normalized;
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
