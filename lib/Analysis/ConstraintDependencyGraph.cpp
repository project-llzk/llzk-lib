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
#include "llzk/Dialect/Constrain/IR/Ops.h"
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

#include <map>
#include <numeric>

#define DEBUG_TYPE "llzk-cdg"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace constrain;
using namespace function;
using namespace pod;

namespace {

static bool isInMaybeSkippedScfRegion(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    if (llvm::isa<FuncDefOp>(parent)) {
      return false;
    }
    if (llvm::isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp>(parent)) {
      return true;
    }
  }
  return false;
}

/// Return whether `address` is rooted at storage allocated anew within `loop`.
/// Such storage cannot carry a write from a previous iteration to the current one.
static bool isAllocatedWithinLoop(const SourceRef &address, Operation *loop) {
  auto root = address.getRoot();
  if (failed(root)) {
    return false;
  }
  auto result = llvm::dyn_cast<OpResult>(*root);
  if (!result || !llvm::isa<CreateArrayOp, NewPodOp>(result.getOwner())) {
    return false;
  }
  return loop->isAncestor(result.getOwner());
}

static inline bool hasRangeIndex(const SourceRef &ref) {
  return llvm::any_of(ref.getPath(), [](const SourceRefIndex &index) {
    return index.isIndexRange();
  });
}

/// Return `ref` with one path component replaced, preserving the suffix after it.
static FailureOr<SourceRef>
replacePathIndex(const SourceRef &ref, size_t index, const SourceRefIndex &replacement) {
  ensure(index < ref.getPath().size(), "SourceRef path index is out of bounds");
  SourceRef result = ref;
  for (size_t i = index; i < ref.getPath().size(); ++i) {
    auto parent = result.getParentPrefix();
    ensure(succeeded(parent), "could not get SourceRef parent while replacing path index");
    result = *parent;
  }
  auto child = result.createChild(replacement);
  ensure(succeeded(child), "could not create SourceRef replacement path index");
  result = *child;
  for (const SourceRefIndex &suffix : ref.getPath().drop_front(index + 1)) {
    child = result.createChild(suffix);
    ensure(succeeded(child), "could not restore SourceRef suffix after replacing path index");
    result = *child;
  }
  return result;
}

/// Return the portions of a ranged address not covered by an overlapping point address.
///
/// The result is a disjoint set of range "slabs": each ranged dimension is split around the
/// point after earlier ranged dimensions have been fixed to it. This preserves every address in
/// `rangeAddress` except `pointAddress` without retaining an entry that overlaps the point.
static SmallVector<SourceRef>
subtractPointFromRange(const SourceRef &rangeAddress, const SourceRef &pointAddress) {
  ensure(rangeAddress.overlaps(pointAddress), "point must overlap range before subtraction");
  ensure(!hasRangeIndex(pointAddress), "range subtraction requires a point address");

  SmallVector<SourceRef> result;
  const auto rangePath = rangeAddress.getPath();
  const auto pointPath = pointAddress.getPath();
  ensure(rangePath.size() == pointPath.size(), "overlapping SourceRefs must have matching paths");
  for (auto [index, rangeIndex] : llvm::enumerate(rangePath)) {
    if (!rangeIndex.isIndexRange()) {
      continue;
    }
    ensure(pointPath[index].isIndex(), "point address must use concrete array indices");

    // Partition in lexicographic dimension order. Earlier ranged dimensions have already
    // excluded their point, while later dimensions must retain their full ranges; narrowing
    // every dimension to the point would omit residual cells such as [1, 1] when subtracting
    // [0, 0] from [0, 2) x [0, 2).
    SourceRef slab = rangeAddress;
    for (size_t earlier = 0; earlier < index; ++earlier) {
      if (rangePath[earlier].isIndexRange()) {
        slab = *replacePathIndex(slab, earlier, pointPath[earlier]);
      }
    }
    const auto [low, high] = rangeIndex.getIndexRange();
    const auto point = pointPath[index].getIndex();
    if (low < point) {
      result.push_back(*replacePathIndex(slab, index, SourceRefIndex({low, point})));
    }
    const DynamicAPInt afterPoint = point + 1;
    if (afterPoint < high) {
      result.push_back(*replacePathIndex(slab, index, SourceRefIndex({afterPoint, high})));
    }
  }
  return result;
}

static bool isNonSingletonArrayWriteTarget(const SourceRefLatticeValue &writeTargets) {
  if (writeTargets.isArray()) {
    return llvm::any_of(
        llvm::seq<size_t>(0, writeTargets.getArraySize()), [&writeTargets](size_t i) {
      return isNonSingletonArrayWriteTarget(writeTargets.getElemFlatIdx(i));
    }
    );
  }
  return !writeTargets.isSingleValue() ||
         llvm::any_of(writeTargets.getScalarValue(), hasRangeIndex);
}

/// Return the point address of a statically shaped array element at `flatIndex`.
static SourceRef getArrayElementAddress(const SourceRef &root, size_t flatIndex) {
  auto arrayType = llvm::dyn_cast<ArrayType>(root.getType());
  ensure(arrayType && arrayType.hasStaticShape(), "array element address requires static shape");
  ArrayIndexGen indexGen = ArrayIndexGen::from(arrayType);
  auto indices = indexGen.delinearize(checkedCast<int64_t>(flatIndex), root.getType().getContext());
  ensure(indices.has_value(), "could not delinearize array element address");
  SourceRef address = root;
  for (Attribute attr : *indices) {
    auto child = address.createChild(SourceRefIndex(llvm::cast<IntegerAttr>(attr).getValue()));
    ensure(succeeded(child), "could not create array element address");
    address = *child;
  }
  return address;
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
    if (top == nullptr) {
      op->walk([this](Operation *nested) {
        operationOrder.try_emplace(nested, operations.size());
        operations.push_back(nested);
      });
      checkpoints.try_emplace(0);
    }
    top = op;
  }

  /// Resolve known storage writes transitively, preserving unwritten and cyclic addresses.
  SourceRefLatticeValue
  resolveDependencies(const SourceRefLatticeValue &addresses, Operation *before) const;

  /// Record a storage write for later dependency queries.
  void recordStorageWrite(
      Operation *op, size_t writeIndex, const SourceRefLatticeValue &addresses,
      const SourceRefLatticeValue &value, bool mayBeSkipped = false,
      bool seedUnwrittenAlternative = false
  );

  /// Rebase an allocation/call result to the aggregate storage receiving it.
  void recordAggregateAlias(
      Operation *op, size_t aliasIndex, const SourceRefLatticeValue &source,
      const SourceRefLatticeValue &target, bool mayBeSkipped = false
  );

  /// Record the storage writes a defined callee applies at a particular call site.
  void recordCalleeStorageWrites(
      CallOpInterface call, FuncDefOp callee, const TranslationMap &translation
  );

  void print(raw_ostream &os) const override { os << "SourceRefAnalysis::StorageState"; }

private:
  struct StorageWrite {
    SourceRefLatticeValue addresses;
    SourceRefLatticeValue value;
    bool mayBeSkipped;
    bool seedUnwrittenAlternative;
  };

  struct AggregateAlias {
    SourceRefLatticeValue source;
    SourceRefLatticeValue target;
    bool mayBeSkipped;
  };

  // NOLINTNEXTLINE(bugprone-exception-escape)
  struct MaterializedStorage {
    DenseMap<SourceRef, SourceRefLatticeValue> values;
    DenseSet<SourceRef> skippedFirstWrites;

    /// Addresses in `values`, grouped by their storage root for dependency lookups.
    DenseMap<Value, SmallVector<SourceRef>> valuesByRoot;

    void materializeOperation(
        const StorageState &state, Operation *op, TranslationMap &aliases, bool forceMayBeSkipped,
        Operation *replayedLoop = nullptr
    );

  private:
    /// Add or remove an address from the root index when `values` changes.
    void indexAddress(const SourceRef &address);
    void unindexAddress(const SourceRef &address);
    void eraseValue(const SourceRef &address);

    void invalidateOverlappingRanges(const SourceRef &address);

    void applyWrite(
        const SourceRef &address, const SourceRefLatticeValue &value, bool maySkip,
        bool seedUnwrittenAlternative
    );

    void materializeWrite(
        const SourceRef &address, const SourceRefLatticeValue &value, bool maySkip,
        bool seedUnwrittenAlternative, bool resolveSelfReference, const TranslationMap &aliases
    );

    void materializeAddressedWrite(
        const SourceRefLatticeValue &addresses, const SourceRefLatticeValue &value, bool maySkip,
        bool seedUnwrittenAlternative, const TranslationMap &aliases, Operation *replayedLoop
    );
  };

  /// Storage and alias state immediately before an operation in preorder.
  ///
  /// Snapshots contain only the ordinary program-order replay. Loop backedge
  /// effects are query-specific and are applied after retrieving a snapshot.
  struct MaterializedSnapshot {
    TranslationMap aliases;
    MaterializedStorage storage;
  };

  /// Bound replay work while avoiding a full state copy at every query point.
  static constexpr size_t checkpointStride = 64;

  /// Apply all known aggregate-storage aliases to a lattice value.
  static SourceRefLatticeValue
  canonicalize(const SourceRefLatticeValue &value, const TranslationMap &aliases);

  /// Apply aggregate aliases recorded for `op` to `aliases`.
  void applyAggregateAliases(
      Operation *op, TranslationMap &aliases, bool forceMayBeSkipped = false
  ) const;

  /// Materialize storage and aliases at `before`, replaying from a sparse checkpoint.
  MaterializedSnapshot materializeSnapshot(Operation *before) const;

  /// Discard checkpoints whose program-order replay includes `op`.
  void invalidateCheckpointsFrom(Operation *op);

  static SourceRefLatticeValue projectChild(
      const SourceRefLatticeValue &value, const SourceRef &storedAddress,
      const SourceRef &readAddress
  );

  SourceRefLatticeValue resolve(
      const SourceRefLatticeValue &input, const TranslationMap &aliases,
      const MaterializedStorage &storage, DenseSet<SourceRef> &active
  ) const;

  void recordCalleeStorageWritesImpl(
      Operation *op, const TranslationMap &translation,
      SmallVectorImpl<StorageWrite> &translatedWrites, bool callMayBeSkipped
  ) const;

  /// Translate caller-visible aggregate aliases from a callee operation.
  void recordCalleeAggregateAliasesImpl(
      Operation *op, const TranslationMap &translation,
      SmallVectorImpl<AggregateAlias> &translatedAliases, bool callMayBeSkipped
  ) const;

  Operation *top = nullptr;
  DenseMap<Operation *, SmallVector<StorageWrite>> storageWrites;
  DenseMap<Operation *, SmallVector<AggregateAlias>> aggregateAliases;
  DenseMap<Operation *, size_t> operationOrder;
  SmallVector<Operation *> operations;
  mutable std::map<size_t, MaterializedSnapshot> checkpoints;
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
  // Region-branch result lattices merge the references carried by their exiting regions, but a
  // storage-backed yielded value must be resolved at the region terminator rather than before
  // the enclosing operation. Resolving it before the region branch would omit writes performed
  // earlier in the region. Repetitive region branches retain the existing loop-carried model.
  if (auto result = llvm::dyn_cast<OpResult>(val)) {
    auto regionBranch = llvm::dyn_cast<RegionBranchOpInterface>(result.getOwner());
    if (regionBranch && !regionBranch.hasLoop()) {
      SourceRefLatticeValue yieldedDependencies;
      bool foundYield = false;
      const unsigned resultNumber = result.getResultNumber();
      for (Region &region : regionBranch->getRegions()) {
        SmallVector<RegionSuccessor> successors;
        regionBranch.getSuccessorRegions(RegionBranchPoint(&region), successors);
        const bool exitsToParent = llvm::any_of(successors, [](const RegionSuccessor &successor) {
          return successor.isParent();
        });
        if (!exitsToParent) {
          continue;
        }
        for (Block &block : region) {
          Operation *terminator = block.getTerminator();
          if (terminator == nullptr) {
            continue;
          }
          auto resolveYieldedOperand = [&](ValueRange yieldedOperands) {
            if (resultNumber >= yieldedOperands.size()) {
              return;
            }
            (void)yieldedDependencies.update(
                getDependencyState(solver, yieldedOperands[resultNumber], terminator)
            );
            foundYield = true;
          };
          if (auto branch = llvm::dyn_cast<RegionBranchTerminatorOpInterface>(terminator)) {
            resolveYieldedOperand(branch.getSuccessorOperands(RegionBranchPoint::parent()));
          } else if (terminator->hasTrait<OpTrait::ReturnLike>()) {
            resolveYieldedOperand(terminator->getOperands());
          }
        }
      }
      if (foundYield) {
        return yieldedDependencies;
      }
    }
  }

  Operation *before = val.getDefiningOp();
  if (before == nullptr) {
    before = val.getParentBlock()->getParentOp();
  }
  return getDependencyState(solver, val, before);
}

SourceRefLatticeValue
SourceRefAnalysis::getDependencyState(DataFlowSolver &solver, Value val, Operation *before) {
  return getDependencyState(solver, getValueState(solver, val), before);
}

SourceRefLatticeValue SourceRefAnalysis::getDependencyState(
    DataFlowSolver &solver, const SourceRefLatticeValue &refs, Operation *before
) {
  Operation *top = before;
  while (top->getParentOp() != nullptr) {
    top = top->getParentOp();
  }
  if (const auto *state = solver.lookupState<StorageState>(solver.getProgramPointBefore(top))) {
    return state->resolveDependencies(refs, before);
  }
  return refs;
}

SourceRefAnalysis::StorageState *SourceRefAnalysis::getStorageState(Operation *op) {
  while (op->getParentOp() != nullptr) {
    op = op->getParentOp();
  }
  auto *state = getOrCreate<StorageState>(getProgramPointBefore(op));
  state->setTop(op);
  return state;
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

SourceRefLatticeValue SourceRefAnalysis::StorageState::projectChild(
    const SourceRefLatticeValue &value, const SourceRef &storedAddress, const SourceRef &readAddress
) {
  if (!readAddress.isValidPrefix(storedAddress) && !readAddress.overlaps(storedAddress)) {
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
    if (ref == storedAddress && hasRangeIndex(ref)) {
      (void)result.insert(ref.narrowRanges(readAddress));
    } else if (auto translated = readAddress.translate(storedAddress, ref); succeeded(translated)) {
      (void)result.insert(*translated);
    } else {
      (void)result.insert(ref);
    }
  }
  return result;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolve(
    const SourceRefLatticeValue &input, const TranslationMap &aliases,
    const MaterializedStorage &storage, DenseSet<SourceRef> &active
) const {
  SourceRefLatticeValue addressValue = canonicalize(input, aliases);
  if (addressValue.isArray()) {
    SourceRefLatticeValue result(addressValue.getArrayShape());
    for (size_t i = 0; i < addressValue.getArraySize(); ++i) {
      (void)result.getElemFlatIdx(i).setValue(
          resolve(addressValue.getElemFlatIdx(i), aliases, storage, active)
      );
    }
    return result;
  }

  SourceRefLatticeValue result;
  for (const SourceRef &address : addressValue.getScalarValue()) {
    if (!active.insert(address).second) {
      if (addressValue.isSingleValue() || storage.skippedFirstWrites.contains(address)) {
        // A skipped first write includes its destination as the unwritten alternative. Preserve
        // that alternative even when the write value contributes other dependencies.
        (void)result.insert(address);
      }
      continue;
    }

    bool foundWrite = false;
    bool preservesAddress = false;
    SourceRefLatticeValue writtenValues;
    auto addressRoot = address.getRoot();
    if (succeeded(addressRoot)) {
      auto writesForRoot = storage.valuesByRoot.find(*addressRoot);
      if (writesForRoot != storage.valuesByRoot.end()) {
        for (const SourceRef &storedAddress : writesForRoot->second) {
          auto storedValue = storage.values.find(storedAddress);
          ensure(storedValue != storage.values.end(), "storage root index is out of sync");
          if (!storedAddress.overlaps(address) && !address.isValidPrefix(storedAddress)) {
            continue;
          }
          foundWrite = true;
          SourceRefLatticeValue projectedValue =
              projectChild(storedValue->second, storedAddress, address);
          if (hasRangeIndex(storedAddress) && storedAddress.overlaps(address)) {
            preservesAddress |= projectedValue.remove(address) == ChangeResult::Change;
          }
          (void)writtenValues.update(projectedValue);
        }
      }
    }
    if (foundWrite) {
      SourceRefLatticeValue resolvedValues = resolve(writtenValues, aliases, storage, active);
      (void)result.update(resolvedValues);
      if (preservesAddress) {
        (void)result.insert(address);
      }
    }
    if (!foundWrite) {
      (void)result.insert(address);
    }
    active.erase(address);
  }
  return result;
}

SourceRefLatticeValue SourceRefAnalysis::StorageState::resolveDependencies(
    const SourceRefLatticeValue &addresses, Operation *before
) const {
  MaterializedSnapshot snapshot = materializeSnapshot(before);
  DenseSet<SourceRef> active;
  return resolve(addresses, snapshot.aliases, snapshot.storage, active);
}

void SourceRefAnalysis::StorageState::recordStorageWrite(
    Operation *op, size_t writeIndex, const SourceRefLatticeValue &addresses,
    const SourceRefLatticeValue &value, bool mayBeSkipped, bool seedUnwrittenAlternative
) {
  invalidateCheckpointsFrom(op);
  auto &writes = storageWrites[op];
  // A scalar storage-backed RHS denotes the contents observed before this write. Resolve it at
  // the write point so a later overwrite cannot change an already-consumed value during replay.
  // Aggregate values retain their storage identity: assignments of those values are modeled by
  // aggregate aliases and must continue to observe subsequent writes through the alias.
  const bool isAggregateValue =
      !op->getOperands().empty() &&
      llvm::isa<ArrayType, StructType, PodType>(op->getOperands().back().getType());
  StorageWrite write {
      addresses, isAggregateValue ? value : resolveDependencies(value, op), mayBeSkipped,
      seedUnwrittenAlternative
  };
  if (writeIndex == writes.size()) {
    writes.push_back(std::move(write));
    return;
  }
  ensure(writeIndex < writes.size(), "storage writes must be recorded in stable operation order");
  writes[writeIndex] = std::move(write);
}

void SourceRefAnalysis::StorageState::recordCalleeStorageWritesImpl(
    Operation *op, const TranslationMap &translation,
    SmallVectorImpl<StorageWrite> &translatedWrites, bool callMayBeSkipped
) const {
  auto writes = storageWrites.find(op);
  if (writes == storageWrites.end()) {
    return;
  }
  for (const StorageWrite &write : writes->second) {
    // A write whose address is not rooted in a callee argument is local to the
    // callee. `translate` drops those addresses, while retaining only effects
    // visible to its caller.
    auto [addresses, addressChange] = write.addresses.translate(translation);
    if (addressChange == ChangeResult::NoChange || addresses.foldToScalar().empty()) {
      continue;
    }
    // Resolve storage-backed values at the callee write point before translating its arguments.
    // For example, a value read from a locally-created POD is initially rooted in that POD, but
    // resolves to the argument used to initialize the record. `replacePrefixes` alone cannot
    // translate that callee-local root.
    SourceRefLatticeValue resolvedValue = resolveDependencies(write.value, op);
    // Unlike addresses, values may legitimately include constants. Keep such
    // sources while replacing every reference rooted in a callee argument.
    auto [value, _] = resolvedValue.replacePrefixes(translation);
    const bool translatedWriteMayBeSkipped =
        callMayBeSkipped || isNonSingletonArrayWriteTarget(addresses);
    translatedWrites.push_back({
        std::move(addresses),
        std::move(value),
        write.mayBeSkipped || translatedWriteMayBeSkipped,
        write.seedUnwrittenAlternative || translatedWriteMayBeSkipped,
    });
  }
}

void SourceRefAnalysis::StorageState::recordCalleeAggregateAliasesImpl(
    Operation *op, const TranslationMap &translation,
    SmallVectorImpl<AggregateAlias> &translatedAliases, bool callMayBeSkipped
) const {
  auto aliases = aggregateAliases.find(op);
  if (aliases == aggregateAliases.end()) {
    return;
  }
  for (const AggregateAlias &alias : aliases->second) {
    // Both sides must be visible to the caller. An alias rooted in callee-local
    // storage only models the callee's temporary state and cannot observe a
    // later caller write through its source.
    auto [source, sourceChange] = alias.source.translate(translation);
    auto [target, targetChange] = alias.target.translate(translation);
    if (sourceChange == ChangeResult::NoChange || targetChange == ChangeResult::NoChange ||
        source.foldToScalar().empty() || target.foldToScalar().empty()) {
      continue;
    }
    translatedAliases.push_back({
        std::move(source),
        std::move(target),
        alias.mayBeSkipped || callMayBeSkipped,
    });
  }
}

void SourceRefAnalysis::StorageState::recordCalleeStorageWrites(
    CallOpInterface call, FuncDefOp callee, const TranslationMap &translation
) {
  invalidateCheckpointsFrom(call.getOperation());
  SmallVector<StorageWrite> translatedWrites;
  SmallVector<AggregateAlias> translatedAliases;
  const bool callMayBeSkipped = isInMaybeSkippedScfRegion(call.getOperation());
  if (Region *callableRegion = callee.getCallableRegion()) {
    for (Block &block : callableRegion->getBlocks()) {
      for (Operation &op : block.getOperations()) {
        recordCalleeStorageWritesImpl(&op, translation, translatedWrites, callMayBeSkipped);
        recordCalleeAggregateAliasesImpl(&op, translation, translatedAliases, callMayBeSkipped);
      }
    }
  }

  // Recompute the complete summary on every solver revisit. This prevents
  // duplicate events and lets operand lattice refinements update the call-site
  // translation without changing its position in program order.
  storageWrites[call.getOperation()] = std::move(translatedWrites);
  aggregateAliases[call.getOperation()] = std::move(translatedAliases);
}

SourceRefAnalysis::StorageState::MaterializedSnapshot
SourceRefAnalysis::StorageState::materializeSnapshot(Operation *before) const {
  ensure(top != nullptr, "storage state must be associated with a top-level operation");

  auto target = operationOrder.find(before);
  ensure(target != operationOrder.end(), "snapshot operation must belong to storage state");

  // Materialize full checkpoints only at fixed program-order intervals. A
  // checkpoint represents state before its key operation, so replay starts at
  // that operation.
  MaterializedSnapshot snapshot;
  size_t replayBegin = 0;
  const size_t checkpointTarget = target->second / checkpointStride * checkpointStride;
  auto nextCheckpoint = checkpoints.upper_bound(checkpointTarget);
  if (nextCheckpoint != checkpoints.begin()) {
    auto previousCheckpoint = std::prev(nextCheckpoint);
    snapshot = previousCheckpoint->second;
    replayBegin = previousCheckpoint->first;
  }

  // Build any missing checkpoints on the way to the one preceding this query.
  // Their immutable program-order state can be reused by later queries.
  for (size_t i = replayBegin; i < checkpointTarget; ++i) {
    snapshot.storage.materializeOperation(
        *this, operations[i], snapshot.aliases, /*forceMayBeSkipped=*/false
    );
    if ((i + 1) % checkpointStride == 0) {
      checkpoints.try_emplace(i + 1, snapshot);
    }
  }

  // The remaining replay is bounded by the checkpoint stride.
  for (size_t i = checkpointTarget; i < target->second; ++i) {
    snapshot.storage.materializeOperation(
        *this, operations[i], snapshot.aliases, /*forceMayBeSkipped=*/false
    );
  }

  // A read in a loop body can observe a write lexically after it from a prior iteration. Replay
  // every enclosing loop body as a weak update: the entry state remains possible (the loop may
  // execute zero times), while its writes form the loop backedge alternative.
  for (Operation *ancestor = before->getParentOp(); ancestor != nullptr;
       ancestor = ancestor->getParentOp()) {
    if (!llvm::isa<scf::ForOp, scf::WhileOp>(ancestor)) {
      continue;
    }
    (void)ancestor->walk([&ancestor, &snapshot, this](Operation *op) {
      if (op != ancestor) {
        snapshot.storage.materializeOperation(
            *this, op, snapshot.aliases, /*forceMayBeSkipped=*/true, ancestor
        );
      }
    });
  }
  return snapshot;
}

void SourceRefAnalysis::StorageState::invalidateCheckpointsFrom(Operation *op) {
  auto position = operationOrder.find(op);
  if (position == operationOrder.end()) {
    return;
  }
  // A checkpoint immediately before this operation excludes its effects, while
  // every later checkpoint includes them and must be reconstructed on a revisit.
  checkpoints.erase(checkpoints.upper_bound(position->second), checkpoints.end());
}

void SourceRefAnalysis::StorageState::MaterializedStorage::indexAddress(const SourceRef &address) {
  if (auto root = address.getRoot(); succeeded(root)) {
    valuesByRoot[*root].push_back(address);
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::unindexAddress(
    const SourceRef &address
) {
  auto root = address.getRoot();
  if (failed(root)) {
    return;
  }
  auto entries = valuesByRoot.find(*root);
  ensure(entries != valuesByRoot.end(), "storage root index is out of sync");
  auto *addressIt = llvm::find(entries->second, address);
  ensure(addressIt != entries->second.end(), "storage root index is out of sync");
  entries->second.erase(addressIt);
  if (entries->second.empty()) {
    valuesByRoot.erase(entries);
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::eraseValue(const SourceRef &address) {
  unindexAddress(address);
  values.erase(address);
}

void SourceRefAnalysis::StorageState::MaterializedStorage::invalidateOverlappingRanges(
    const SourceRef &address
) {
  // A definite point write supersedes an earlier weak range write at that point. Keep the
  // portions of the range that can still affect other reads, but remove its overlap with this
  // address so dependency resolution does not join stale values at the overwritten element.
  struct RangeReplacement {
    SourceRef address;
    SourceRefLatticeValue value;
    bool wasSkippedFirstWrite;
  };

  SmallVector<RangeReplacement> replacements;
  SmallVector<SourceRef> erasedAddresses;
  for (const auto &[storedAddress, storedValue] : values) {
    if (!hasRangeIndex(storedAddress) || !storedAddress.overlaps(address)) {
      continue;
    }
    erasedAddresses.push_back(storedAddress);
    for (const SourceRef &residual : subtractPointFromRange(storedAddress, address)) {
      // A weak write may carry its destination as the unwritten alternative. Rebase that
      // alternative to the residual range too; otherwise a later read would retain the
      // original, still-overlapping range through the value rather than the map key.
      auto [residualValue, _] = storedValue.replacePrefixes(
          TranslationMap {{storedAddress, SourceRefLatticeValue(residual)}}
      );
      replacements.push_back(
          {residual, std::move(residualValue), skippedFirstWrites.contains(storedAddress)}
      );
    }
  }
  for (const SourceRef &erasedAddress : erasedAddresses) {
    skippedFirstWrites.erase(erasedAddress);
    eraseValue(erasedAddress);
  }
  for (const RangeReplacement &replacement : replacements) {
    auto [it, inserted] = values.try_emplace(replacement.address, replacement.value);
    if (inserted) {
      indexAddress(replacement.address);
    }
    if (!inserted) {
      (void)it->second.update(replacement.value);
    }
    if (replacement.wasSkippedFirstWrite) {
      skippedFirstWrites.insert(replacement.address);
    }
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::applyWrite(
    const SourceRef &address, const SourceRefLatticeValue &value, bool maySkip,
    bool seedUnwrittenAlternative
) {
  // A skipped first write can leave the storage unwritten. Seed the stored state with both
  // alternatives so reads retain the address dependency along that path.
  SourceRefLatticeValue initialValue = value;
  if (seedUnwrittenAlternative) {
    (void)initialValue.insert(address);
  }
  auto [it, inserted] = values.try_emplace(address, std::move(initialValue));
  if (inserted) {
    indexAddress(address);
  }
  if (inserted && seedUnwrittenAlternative) {
    skippedFirstWrites.insert(address);
  }
  if (!inserted) {
    if (maySkip) {
      (void)it->second.update(value);
    } else {
      (void)it->second.setValue(value);
    }
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::materializeWrite(
    const SourceRef &address, const SourceRefLatticeValue &value, bool maySkip,
    bool seedUnwrittenAlternative, bool resolveSelfReference, const TranslationMap &aliases
) {
  SourceRefLatticeValue canonicalValue = StorageState::canonicalize(value, aliases);
  auto arrayType = llvm::dyn_cast<ArrayType>(address.getType());
  if (canonicalValue.isArray() && arrayType && arrayType.hasStaticShape() &&
      std::cmp_equal(canonicalValue.getArraySize(), arrayType.getNumElements())) {
    for (size_t i = 0; i < canonicalValue.getArraySize(); ++i) {
      materializeWrite(
          getArrayElementAddress(address, i), canonicalValue.getElemFlatIdx(i), maySkip,
          seedUnwrittenAlternative, resolveSelfReference, aliases
      );
    }
    return;
  }
  if (canonicalValue.isScalar() && canonicalValue.getScalarValue().contains(address)) {
    if (auto preWrite = values.find(address); resolveSelfReference && preWrite != values.end()) {
      // The self-reference denotes the value read before this write. Substitute the
      // materialized pre-write contents so a read-modify-write retains those dependencies.
      (void)canonicalValue.getScalarValue().erase(address);
      (void)canonicalValue.update(preWrite->second);
    } else {
      const bool hasPriorContents = llvm::any_of(values, [&address](const auto &entry) {
        return entry.first.isValidPrefix(address);
      });
      if (hasPriorContents || canonicalValue.isSingleValue()) {
        (void)canonicalValue.getScalarValue().erase(address);
        if (canonicalValue.getScalarValue().empty()) {
          return;
        }
      }
    }
  }
  if (!maySkip && !hasRangeIndex(address)) {
    invalidateOverlappingRanges(address);
  }
  applyWrite(address, canonicalValue, maySkip, seedUnwrittenAlternative);
}

void SourceRefAnalysis::StorageState::MaterializedStorage::materializeAddressedWrite(
    const SourceRefLatticeValue &addresses, const SourceRefLatticeValue &value, bool maySkip,
    bool seedUnwrittenAlternative, const TranslationMap &aliases, Operation *replayedLoop
) {
  SourceRefLatticeValue canonicalAddresses = StorageState::canonicalize(addresses, aliases);
  if (canonicalAddresses.isArray() && value.isArray() &&
      canonicalAddresses.getArraySize() == value.getArraySize()) {
    for (size_t i = 0; i < canonicalAddresses.getArraySize(); ++i) {
      materializeAddressedWrite(
          canonicalAddresses.getElemFlatIdx(i), value.getElemFlatIdx(i), maySkip,
          seedUnwrittenAlternative, aliases, replayedLoop
      );
    }
    return;
  }
  for (const SourceRef &address : canonicalAddresses.foldToScalar()) {
    if (replayedLoop && isAllocatedWithinLoop(address, replayedLoop)) {
      continue;
    }
    // Canonicalization can expand one write address into several possible alias targets. Each
    // target must retain its old contents because the write affects only one at runtime.
    const bool expandedAliasTargets = !canonicalAddresses.isSingleValue();
    materializeWrite(
        address, value, maySkip || expandedAliasTargets,
        seedUnwrittenAlternative || expandedAliasTargets,
        /*resolveSelfReference=*/true, aliases
    );
  }
}

void SourceRefAnalysis::StorageState::MaterializedStorage::materializeOperation(
    const StorageState &state, Operation *op, TranslationMap &aliases, bool forceMayBeSkipped,
    Operation *replayedLoop
) {
  DenseSet<SourceRef> priorAliasSources;
  for (const auto &[source, _] : aliases) {
    priorAliasSources.insert(source);
  }
  state.applyAggregateAliases(op, aliases, forceMayBeSkipped);

  // An aggregate assignment transfers the source's current contents to its new storage
  // location. Rebase only aliases introduced by this operation: applying the complete alias
  // map here would incorrectly move older writes in response to later assignments.
  SmallVector<std::pair<SourceRef, SourceRefLatticeValue>> newAliases;
  for (const auto &[source, targets] : aliases) {
    if (!priorAliasSources.contains(source)) {
      newAliases.emplace_back(source, targets);
    }
  }
  for (const auto &[source, targets] : newAliases) {
    const bool mayBeSkipped =
        forceMayBeSkipped || (targets.isScalar() && targets.getScalarValue().contains(source));
    SmallVector<std::pair<SourceRef, SourceRefLatticeValue>> sourceContents;
    for (const auto &[address, value] : values) {
      if (address.isValidPrefix(source)) {
        sourceContents.emplace_back(address, value);
      }
    }
    for (const auto &[address, value] : sourceContents) {
      for (const SourceRef &target : targets.foldToScalar()) {
        if (replayedLoop && isAllocatedWithinLoop(target, replayedLoop)) {
          continue;
        }
        auto rebasedAddress = address.translate(source, target);
        if (succeeded(rebasedAddress)) {
          materializeWrite(
              *rebasedAddress, value, mayBeSkipped,
              /*seedUnwrittenAlternative=*/false,
              /*resolveSelfReference=*/false, aliases
          );
        }
      }
    }
  }

  auto writes = state.storageWrites.find(op);
  if (writes == state.storageWrites.end()) {
    return;
  }
  for (const StorageWrite &write : writes->second) {
    materializeAddressedWrite(
        write.addresses, write.value, forceMayBeSkipped || write.mayBeSkipped,
        forceMayBeSkipped || write.seedUnwrittenAlternative, aliases, replayedLoop
    );
  }
}

void SourceRefAnalysis::StorageState::recordAggregateAlias(
    Operation *op, size_t aliasIndex, const SourceRefLatticeValue &source,
    const SourceRefLatticeValue &target, bool mayBeSkipped
) {
  invalidateCheckpointsFrom(op);
  auto &aliases = aggregateAliases[op];
  AggregateAlias alias {source, target, mayBeSkipped};
  if (aliasIndex == aliases.size()) {
    aliases.push_back(std::move(alias));
    return;
  }
  ensure(aliasIndex < aliases.size(), "aggregate aliases must use stable operation order");
  aliases[aliasIndex] = std::move(alias);
}

void SourceRefAnalysis::StorageState::applyAggregateAliases(
    Operation *op, TranslationMap &aliases, bool forceMayBeSkipped
) const {
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
            SourceRefLatticeValue(getArrayElementAddress(targetRoot, i)), mayBeSkipped
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
      aliases.set(sourceRef, std::move(aliasTargets));
    }
  };

  auto events = aggregateAliases.find(op);
  if (events != aggregateAliases.end()) {
    for (const AggregateAlias &event : events->second) {
      addAlias(event.source, event.target, forceMayBeSkipped || event.mayBeSkipped);
    }
  }
}

FailureOr<SourceRefLatticeValue>
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

  return failure();
}

void SourceRefAnalysis::setToEntryState(Lattice *lattice) {
  if (auto value = llvm::dyn_cast_if_present<Value>(lattice->getAnchor())) {
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
    } else {
      auto [memberVals, _] = *memberValsRes;
      auto writeOp = llvm::cast<MemberWriteOp>(op);
      SourceRefLatticeValue writeValue = operandVals.at(writeOp.getVal())->getValue();
      if (llvm::isa<ArrayType, StructType, PodType>(writeOp.getVal().getType())) {
        // A component value that merges multiple storage roots selects exactly
        // one root at runtime. Model the write as weak for every candidate so
        // the unselected roots retain their pre-write contents.
        const bool mayBeSkipped = isInMaybeSkippedScfRegion(op) || !memberVals.isSingleValue();
        getStorageState(op)->recordStorageWrite(
            op, /*writeIndex=*/0, memberVals, writeValue, mayBeSkipped,
            /*seedUnwrittenAlternative=*/mayBeSkipped
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
      // A POD value that merges multiple storage roots selects exactly one root at runtime.
      // Model the write as weak for every candidate so the unselected roots retain their
      // pre-write contents.
      const bool mayBeSkipped = isInMaybeSkippedScfRegion(op) || !podVals.isSingleValue();
      getStorageState(op)->recordStorageWrite(
          op, /*writeIndex=*/0, podVals, writeValue, mayBeSkipped,
          /*seedUnwrittenAlternative=*/mayBeSkipped
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
    if (llvm::isa<WriteArrayOp, InsertArrayOp>(arrayAccessOp)) {
      SourceRefLatticeValue writeTargets = arraySubdivisionOpUpdate(arrayAccessOp, operandVals);
      Value rvalue = op->getOperands().back();
      SourceRefLatticeValue writeValue = operandVals.at(rvalue)->getValue();
      const bool mayBeSkipped =
          isInMaybeSkippedScfRegion(op) || isNonSingletonArrayWriteTarget(writeTargets);
      getStorageState(op)->recordStorageWrite(
          op, /*writeIndex=*/0, writeTargets, writeValue, mayBeSkipped,
          /*seedUnwrittenAlternative=*/mayBeSkipped
      );
      if (llvm::isa<ArrayType, StructType, PodType>(rvalue.getType())) {
        getStorageState(op)->recordAggregateAlias(
            op, /*aliasIndex=*/0, writeValue, writeTargets, mayBeSkipped
        );
      }
    }
    if (!results.empty()) {
      auto newVals = arraySubdivisionOpUpdate(arrayAccessOp, operandVals);
      propagateIfChanged(results.front(), results.front()->setValue(newVals));
    }
    return success();
  }

  if (auto createArray = llvm::dyn_cast<CreateArrayOp>(op)) {
    auto createArrayRes = createArray.getResult();
    SourceRef arrayRoot(llvm::cast<OpResult>(createArrayRes));
    auto arrayType = createArray.getType();

    if (arrayType.hasStaticShape()) {
      SourceRefLatticeValue newArrayValue(arrayType.getShape());
      for (size_t i = 0; i < static_cast<size_t>(arrayType.getNumElements()); ++i) {
        (void)newArrayValue.getElemFlatIdx(i).setValue(
            SourceRefLatticeValue(getArrayElementAddress(arrayRoot, i))
        );
      }
      propagateIfChanged(results.front(), results.front()->setValue(newArrayValue));
    } else {
      SourceRefLatticeValue newArrayValue(arrayRoot);
      propagateIfChanged(results.front(), results.front()->setValue(newArrayValue));
    }

    const auto &elements = createArray.getElements();
    if (elements.empty()) {
      return success();
    }

    const bool mayBeSkipped = isInMaybeSkippedScfRegion(op);
    StorageState *state = getStorageState(op);
    for (size_t i = 0; i < elements.size(); i++) {
      SourceRef elementAddress = getArrayElementAddress(arrayRoot, i);
      SourceRefLatticeValue elementAddressValue(elementAddress);
      SourceRefLatticeValue elementValue = operandVals.at(elements[i])->getValue();
      state->recordStorageWrite(op, i, elementAddressValue, elementValue, mayBeSkipped);
      if (llvm::isa<ArrayType, StructType, PodType>(elements[i].getType())) {
        state->recordAggregateAlias(op, i, elementValue, elementAddressValue, mayBeSkipped);
      }
    }
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
  // Call is to a defined function with a body, but it's treated as external so we
  // can translate the results based on the arguments.
  auto funcOpRes = resolveCallable<FuncDefOp>(tables, call);
  if (failed(funcOpRes) || !*funcOpRes) {
    // Other callable operations, such as verif.contract, can have a body
    // without defining function storage effects to translate. In particular,
    // resultless calls have no value lattice to update either.
    return;
  }

  TranslationMap translation;
  TranslationMap storageTranslation;
  FuncDefOp funcOp = funcOpRes->get();
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    SourceRefLatticeValue argumentValue =
        static_cast<const Lattice *>(operandLattices[i])->getValue();
    const bool isAggregate =
        llvm::isa<ArrayType, StructType, PodType>(call->getOperand(i).getType());
    if (!isAggregate) {
      argumentValue = getStorageState(call.getOperation())
                          ->resolveDependencies(argumentValue, call.getOperation());
    }
    SourceRef argumentRef(funcOp.getArgument(i));
    translation.set(argumentRef, argumentValue);
    // Aggregate value lattices are shaped by their elements. Storage effects,
    // however, need the aggregate root so a callee selection is appended only
    // once (e.g. `%arg[0]`, never `%arg[0][0]`).
    if (isAggregate) {
      auto storageArgumentRef = SourceRefLattice::getSourceRef(call->getOperand(i));
      if (succeeded(storageArgumentRef)) {
        storageTranslation.set(argumentRef, SourceRefLatticeValue(*storageArgumentRef));
      } else if (argumentValue.isScalar()) {
        // Reads of aggregate POD records or struct members are not rooted
        // constructors. Their raw lattice nevertheless names the selected
        // aggregate, so use it as the storage translation.
        storageTranslation.set(argumentRef, argumentValue);
      } else if (auto arrayType = llvm::dyn_cast<ArrayType>(call->getOperand(i).getType())) {
        // An array.extract of an array-shaped value has one lattice element per
        // selected leaf. Recover the selected subarray address by removing the
        // dimensions belonging to the argument type from every leaf. This keeps
        // the extract path while avoiding a duplicated callee index during
        // translation (e.g. `%matrix[1]` + callee `[0]`, not `%matrix[1][0][0]`).
        SourceRefLatticeValue storageArgumentValue;
        for (const SourceRef &leaf : argumentValue.foldToScalar()) {
          SourceRef selectedAggregate = leaf;
          bool recovered = true;
          for (int64_t dim = 0; dim < arrayType.getRank(); ++dim) {
            auto parent = selectedAggregate.getParentPrefix();
            if (failed(parent)) {
              recovered = false;
              break;
            }
            selectedAggregate = *parent;
          }
          if (recovered) {
            (void)storageArgumentValue.insert(selectedAggregate);
          }
        }
        if (!storageArgumentValue.foldToScalar().empty()) {
          storageTranslation.set(argumentRef, std::move(storageArgumentValue));
        }
      }
    } else {
      storageTranslation.set(argumentRef, argumentValue);
    }
  }
  getStorageState(call.getOperation())->recordCalleeStorageWrites(call, funcOp, storageTranslation);

  if (resultLattices.empty()) {
    // No-result calls can still mutate aggregate arguments. Their translated
    // storage effects have been recorded above, but there are no result values
    // to translate back to the caller.
    return;
  }

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

  for (auto [result, resultLattice] : llvm::zip(call->getResults(), resultLattices)) {
    (void)result;
    SourceRefLatticeValue combined;
    unsigned resultNum = llvm::cast<OpResult>(result).getResultNumber();
    for (Operation *returnSite : returnSites) {
      SourceRefLatticeValue retVal =
          getLatticeElementFor(
              getProgramPointAfter(call.getOperation()), returnSite->getOperand(resultNum)
          )
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
    ArrayAccessOpInterface arrayAccessOp, const OperandValues &operandVals
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
    return failure();
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

LogicalResult
ConstraintDependencyGraph::computeConstraints(DataFlowSolver &solver, AnalysisManager &am) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  FuncDefOp constrainFnOp = structDef.getConstrainFuncOp();
  ensure(
      constrainFnOp,
      "malformed struct " + Twine(structDef.getName()) + " must define a constrain function"
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

  if (!ctx.runIntraproceduralAnalysis()) {
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
      ensure(succeeded(res), "could not resolve constrain call");

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
        SourceRefLatticeValue val;
        if (llvm::isa<ArrayType>(operand.getType())) {
          val = SourceRefAnalysis::getDependencyState(solver, operand, fnCall.getOperation());
        } else if (llvm::isa<StructType, PodType>(operand.getType())) {
          // Aggregate arguments retain their storage identity until a child path
          // is translated below. That complete reference is then resolved at the
          // call site, so a POD record maps to its stored value rather than its
          // local storage address.
          val = SourceRefAnalysis::getValueState(solver, operand);
        } else {
          val = SourceRefAnalysis::getDependencyState(solver, operand);
        }
        translations.push_back({prefix, val});
      }
      auto &childAnalysis =
          am.getChildAnalysis<ConstraintDependencyGraphStructAnalysis>(calledStruct);
      if (!childAnalysis.constructed(ctx)) {
        ensure(
            succeeded(childAnalysis.runAnalysis(solver, am, {.runIntraprocedural = false})),
            "could not construct CDG for child struct"
        );
      }
      auto translatedCDG = childAnalysis.getResult(ctx).translate(
          translations, [&solver, call = fnCall.getOperation()](const SourceRef &ref) {
        return SourceRefAnalysis::getDependencyState(solver, SourceRefLatticeValue(ref), call);
      }
      );
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
    constrainFnOp.walk(fnCallWalker);
  }

  return success();
}

void ConstraintDependencyGraph::walkConstrainOp(DataFlowSolver &solver, Operation *emitOp) {
  std::vector<SourceRef> signalUsages, constUsages;

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = SourceRefAnalysis::getDependencyState(solver, operand);
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

ConstraintDependencyGraph ConstraintDependencyGraph::translate(
    SourceRefRemappings translation,
    const std::function<SourceRefLatticeValue(const SourceRef &)> &resolve
) const {
  ConstraintDependencyGraph res(mod, structDef, ctx);
  auto translate = [&translation,
                    &resolve](const SourceRef &elem) -> FailureOr<std::vector<SourceRef>> {
    std::vector<SourceRef> refs;
    auto appendRef = [&](const SourceRef &ref) {
      if (!resolve) {
        refs.push_back(ref);
        return;
      }
      SourceRefLatticeValue resolved = resolve(ref);
      auto folded = resolved.foldToScalar();
      refs.insert(refs.end(), folded.begin(), folded.end());
    };
    for (auto &[prefix, vals] : translation) {
      if (!elem.isValidPrefix(prefix)) {
        continue;
      }

      if (vals.isArray()) {
        // Try to index into the array
        auto suffix = elem.getSuffix(prefix);
        ensure(succeeded(suffix), "failure is nonsensical, we already checked for valid prefix");

        auto resolvedValsRes = vals.extract(suffix.value());
        ensure(succeeded(resolvedValsRes), "could not create SourceRef child while resolving refs");
        auto [resolvedVals, _] = *resolvedValsRes;
        auto folded = resolvedVals.foldToScalar();
        for (const SourceRef &ref : folded) {
          appendRef(ref);
        }
      } else {
        for (const auto &replacement : vals.getScalarValue()) {
          auto translated = elem.translate(prefix, replacement);
          if (succeeded(translated)) {
            appendRef(translated.value());
          }
        }
      }
    }
    if (refs.empty()) {
      return failure();
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
      if (failed(member)) {
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
  auto currRef = FailureOr<SourceRef>(ref);
  while (succeeded(currRef)) {
    // A dynamic access is represented by a half-open range. Match every concrete element and
    // range that overlaps the queried path, as well as exact references.
    for (auto candidate = signalSets.begin(); candidate != signalSets.end(); ++candidate) {
      const SourceRef &candidateRef = candidate->getData();
      if (!candidateRef.overlaps(*currRef)) {
        continue;
      }
      for (auto it = signalSets.findLeader(candidate); it != signalSets.member_end(); ++it) {
        if (!it->overlaps(ref)) {
          res.insert(*it);
        }
      }
      auto constIt = constantSets.find(candidateRef);
      if (constIt != constantSets.end()) {
        res.insert(constIt->second.begin(), constIt->second.end());
      }
    }
    // Go to parent
    currRef = currRef->getParentPrefix();
  }
  return res;
}

/* ConstraintDependencyGraphStructAnalysis */

LogicalResult ConstraintDependencyGraphStructAnalysis::runAnalysis(
    DataFlowSolver &solver, AnalysisManager &moduleAnalysisManager, const CDGAnalysisContext &ctx
) {
  auto result = ConstraintDependencyGraph::compute(
      getModule(), getStruct(), solver, moduleAnalysisManager, ctx
  );
  if (failed(result)) {
    return failure();
  }
  setResult(ctx, std::move(*result));
  return success();
}

} // namespace llzk
