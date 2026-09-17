//===-- LLZKRedundantReadAndWriteEliminationPass.cpp ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-duplicate-read-write-elim` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/RAM/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Concepts.h"
#include "llzk/Util/EffectHelper.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>
#include <optional>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_REDUNDANTREADANDWRITEELIMINATIONPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;

#define DEBUG_TYPE "llzk-redundant-read-write-pass"

namespace {

/// @brief A reference to a value, represented by an SSA value, an attribute
/// (e.g., a member name or access metadata), or an int (e.g., a constant array index).
class ReferenceID {
public:
  explicit ReferenceID(Value v) {
    // reserved special pointer values for DenseMapInfo
    if (v == llvm::DenseMapInfo<Value>::getEmptyKey() ||
        v == llvm::DenseMapInfo<Value>::getTombstoneKey()) {
      identifier = v;
    } else if (auto constVal = dyn_cast_if_present<FeltConstantOp>(v.getDefiningOp())) {
      identifier = constVal.getValue();
    } else if (auto constIdxVal = dyn_cast_if_present<arith::ConstantIndexOp>(v.getDefiningOp())) {
      identifier = llvm::cast<IntegerAttr>(constIdxVal.getValue()).getValue();
    } else {
      identifier = v;
    }
  }
  explicit ReferenceID(Attribute attr) : identifier(attr) {}
  explicit ReferenceID(const APInt &i) : identifier(i) {}
  explicit ReferenceID(unsigned i) : identifier(APInt(64, i)) {}

  bool isValue() const { return std::holds_alternative<Value>(identifier); }
  bool isAttribute() const { return std::holds_alternative<Attribute>(identifier); }
  bool isConst() const { return std::holds_alternative<APInt>(identifier); }

  Value getValue() const {
    ensure(isValue(), "does not hold Value");
    return std::get<Value>(identifier);
  }

  Attribute getAttribute() const {
    ensure(isAttribute(), "does not hold Attribute");
    return std::get<Attribute>(identifier);
  }

  APInt getConst() const {
    ensure(isConst(), "does not hold const");
    return std::get<APInt>(identifier);
  }

  void print(raw_ostream &os) const {
    if (const auto *v = std::get_if<Value>(&identifier)) {
      if (auto opres = dyn_cast<OpResult>(*v)) {
        os << '%' << opres.getResultNumber();
      } else {
        os << *v;
      }
    } else if (const auto *attr = std::get_if<Attribute>(&identifier)) {
      os << *attr;
    } else {
      os << std::get<APInt>(identifier);
    }
  }

  friend bool operator==(const ReferenceID &lhs, const ReferenceID &rhs) {
    return lhs.identifier == rhs.identifier;
  }

  friend raw_ostream &operator<<(raw_ostream &os, const ReferenceID &id) {
    id.print(os);
    return os;
  }

private:
  /// @brief Three cases:
  /// Attribute: identifier refers to a named member or other access metadata
  /// APInt: identifier refers to a constant index in an array
  /// Value: identifier refers to a dynamic index or access operand
  std::variant<Attribute, APInt, Value> identifier;
};

} // namespace

namespace llvm {

/// @brief Allows ReferenceID to be a DenseMap key.
template <> struct DenseMapInfo<ReferenceID> {
  static ReferenceID getEmptyKey() { return ReferenceID(DenseMapInfo<Value>::getEmptyKey()); }
  static inline ReferenceID getTombstoneKey() {
    return ReferenceID(DenseMapInfo<Value>::getTombstoneKey());
  }
  static unsigned getHashValue(const ReferenceID &r) {
    if (r.isValue()) {
      return hash_value(r.getValue());
    } else if (r.isAttribute()) {
      return hash_value(r.getAttribute());
    }
    return hash_value(r.getConst());
  }
  static bool isEqual(const ReferenceID &lhs, const ReferenceID &rhs) { return lhs == rhs; }
};

} // namespace llvm

namespace {

/// @brief A node in a tree of references that represent known values. A node consists of:
/// - An identifier (e.g., %self)
/// - A stored value (i.e., the allocation site or the value last written to the identifier)
/// - A map of children (e.g., members of a struct or elements of an array).
/// An example:
/// %self -> @arr -> 1 represents %self[@arr][1].
/// %self -> @column -> -1 : index represents a prior-row member access.
///
/// Values not in this tree are unknown, and therefore not subject to read/write
/// elimination until they become known and can be eliminated when redundant operations
/// are performed.
///
/// Children may represent constant access components (member refs, constant indices)
/// or dynamic SSA values. Dynamic children may alias constant siblings, so array
/// writes clear all children for a dynamic index and only dynamic siblings for a
/// constant index.
class ReferenceNode {
public:
  struct AggregateSnapshotSource {
    Value value;
    Operation *write;
  };

  /// The storage path from which an aggregate read copied its result.
  struct AggregateReadOrigin {
    std::shared_ptr<ReferenceNode> storage;
    uint64_t mutationEpoch;
    Operation *read;
  };

  template <typename IdType> static std::shared_ptr<ReferenceNode> create(IdType id, Value v) {
    ReferenceNode n(id, v);
    // Need the move constructor version since constructor is private
    return std::make_shared<ReferenceNode>(std::move(n));
  }

  /// @brief Clone the current node, creating a new shared_ptr from it, optionally
  /// recursively cloning the children (default is true).
  std::shared_ptr<ReferenceNode> clone(bool withChildren = true) const {
    ReferenceNode copy(identifier, storedValue);
    copy.updateLastWrite(lastWrite);
    copy.aggregateMutationEpoch = aggregateMutationEpoch;
    if (withChildren) {
      copy.dynamicChildCount = dynamicChildCount;
      for (const auto &[id, child] : children) {
        copy.children[id] = child->clone(withChildren);
      }
    }
    return std::make_shared<ReferenceNode>(std::move(copy));
  }

  /// @brief Make an independent snapshot of this aggregate for a value-copy boundary.
  ///
  /// The copied tree deliberately does not retain write-removal candidates from the
  /// source allocation: writes through the copy cannot overwrite writes through the
  /// source allocation, or vice versa.
  std::shared_ptr<ReferenceNode> cloneForValueCopy(Value copiedValue) const {
    auto copy = clone();
    copy->identifier = ReferenceID(copiedValue);
    copy->storedValue = copiedValue;
    copy->clearLastWritesInSubtree();
    return copy;
  }

  template <typename IdType>
  std::shared_ptr<ReferenceNode>
  createChild(IdType id, Value storedVal, const std::shared_ptr<ReferenceNode> &valTree = nullptr) {
    std::shared_ptr<ReferenceNode> child = create(id, storedVal);
    child->setCurrentValue(storedVal, valTree);
    if (child->identifier.isValue() && children.find(child->identifier) == children.end()) {
      ++dynamicChildCount;
    }
    children[child->identifier] = child;
    return child;
  }

  /// @brief Find the child with the given ID. Returns nullptr if no such child exists.
  /// @tparam IdType A type convertible into a ReferenceID.
  template <typename IdType> std::shared_ptr<ReferenceNode> getChild(IdType id) const {
    auto it = children.find(ReferenceID(id));
    if (it != children.end()) {
      return it->second;
    }
    return nullptr;
  }

  /// @brief Find the child with the given ID, or create one with the storedVal if no such child
  /// exists.
  /// @tparam IdType A type convertible into a ReferenceID.
  template <typename IdType>
  std::shared_ptr<ReferenceNode> getOrCreateChild(IdType id, Value storedVal = nullptr) {
    auto it = children.find(ReferenceID(id));
    if (it != children.end()) {
      return it->second;
    }
    return createChild(id, storedVal);
  }

  /// @brief Set the last write that updates this node and return the older write
  /// that is being replaced by `writeOp` (or nullptr if there was no prior write).
  Operation *updateLastWrite(Operation *writeOp) {
    Operation *old = lastWrite;
    lastWrite = writeOp;
    return old;
  }

  void clearLastWrite() { lastWrite = nullptr; }

  /// @brief Clear overwrite candidates that a live indexed read may observe.
  ///
  /// A dynamic index may select any non-member child, while a constant index
  /// may also select an existing dynamic child. Aggregate reads clear the
  /// selected subtree because a later access through the result may observe
  /// writes below it.
  /// @param indices Access path from this node to the observed array element or subtree.
  void clearLastWritesObservedBy(ArrayRef<ReferenceID> indices) {
    if (indices.empty()) {
      clearLastWritesInSubtree();
      return;
    }
    clearLastWrite();

    const ReferenceID &index = indices.front();
    ArrayRef<ReferenceID> remaining = indices.drop_front();
    if (!index.isConst()) {
      for (const auto &[id, child] : children) {
        if (!id.isAttribute()) {
          child->clearLastWritesObservedBy(remaining);
        }
      }
      return;
    }

    if (auto it = children.find(index); it != children.end()) {
      it->second->clearLastWritesObservedBy(remaining);
    }
    for (const auto &[id, child] : children) {
      if (id.isValue()) {
        child->clearLastWritesObservedBy(remaining);
      }
    }
  }

  /// @brief Clear overwrite candidates in this node and all descendants.
  ///
  /// Region boundaries use this to prevent tree-state candidates from being
  /// treated as block-local predecessors outside the region that created them.
  void clearLastWritesInSubtree() {
    for (const auto &[_, child] : children) {
      child->clearLastWritesInSubtree();
    }
    clearLastWrite();
  }

  void setCurrentValue(Value v, const std::shared_ptr<ReferenceNode> &valTree = nullptr) {
    storedValue = v;
    reusableAggregateRead.reset();
    aggregateSnapshotSource.reset();
    aggregateReadOrigin.reset();
    if (valTree != nullptr) {
      // Overwrite our current set of children with new children, since we overwrote
      // the stored value.
      children = valTree->children;
      dynamicChildCount = valTree->dynamicChildCount;
    }
  }

  void invalidateChildren() {
    children.clear();
    dynamicChildCount = 0;
  }

  /// @brief Remove dynamic-index children before descending through a constant index.
  ///
  /// A constant-index write can alias an existing dynamic-index child, but not a
  /// different constant-index child.
  void invalidateDynamicChildren() {
    if (dynamicChildCount == 0) {
      return;
    }
    SmallVector<ReferenceID> invalidChildren;
    for (const auto &[id, _] : children) {
      if (id.isValue()) {
        invalidChildren.push_back(id);
      }
    }
    for (const ReferenceID &id : invalidChildren) {
      children.erase(id);
    }
    dynamicChildCount = 0;
  }

  bool invalidateNonIntegerOffsetChildren() {
    SmallVector<ReferenceID> invalidChildren;
    size_t invalidDynamicChildCount = 0;
    for (const auto &[id, _] : children) {
      if (!id.isAttribute() || !isa<IntegerAttr>(id.getAttribute())) {
        invalidChildren.push_back(id);
        if (id.isValue()) {
          ++invalidDynamicChildCount;
        }
      }
    }
    for (const ReferenceID &id : invalidChildren) {
      children.erase(id);
    }
    dynamicChildCount -= invalidDynamicChildCount;
    return !invalidChildren.empty();
  }

  bool isLeaf() const { return children.empty(); }

  Value getStoredValue() const { return storedValue; }

  bool hasStoredValue() const { return storedValue != nullptr; }

  /// @brief Return an immutable aggregate read that may be reused for this access.
  const std::optional<Value> &getReusableAggregateRead() const { return reusableAggregateRead; }

  /// @brief Remember an immutable aggregate read from this access.
  void setReusableAggregateRead(Value value) { reusableAggregateRead = value; }

  /// @brief Return the value and write operation that created this aggregate snapshot.
  const std::optional<AggregateSnapshotSource> &getAggregateSnapshotSource() const {
    return aggregateSnapshotSource;
  }

  /// @brief Record the source value of the aggregate snapshot stored at this access.
  void setAggregateSnapshotSource(Value value, Operation *write) {
    aggregateSnapshotSource = AggregateSnapshotSource {.value = value, .write = write};
  }

  /// @brief Discard the source provenance after a write into this aggregate.
  void clearAggregateSnapshotSource() { aggregateSnapshotSource.reset(); }

  /// Record that this value is a snapshot read from \p storage.
  void setAggregateReadOrigin(const std::shared_ptr<ReferenceNode> &storage, Operation *read) {
    aggregateReadOrigin = AggregateReadOrigin {
        .storage = storage,
        .mutationEpoch = storage->aggregateMutationEpoch,
        .read = read,
    };
  }

  /// Return the storage path from which this aggregate value was read.
  const std::optional<AggregateReadOrigin> &getAggregateReadOrigin() const {
    return aggregateReadOrigin;
  }

  /// Mark this storage path as changed by a write at or below it.
  void markAggregateMutation() { ++aggregateMutationEpoch; }

  uint64_t getAggregateMutationEpoch() const { return aggregateMutationEpoch; }

  void print(raw_ostream &os, int indent = 0) const {
    os.indent(indent) << '[' << identifier;
    if (storedValue != nullptr) {
      os << " => " << storedValue;
    }
    os << ']';
    if (!children.empty()) {
      os << "{\n";
      for (const auto &[_, child] : children) {
        child->print(os, indent + 4);
        os << '\n';
      }
      os.indent(indent) << '}';
    }
  }

  [[maybe_unused]]
  friend raw_ostream &operator<<(raw_ostream &os, const ReferenceNode &r) {
    r.print(os);
    return os;
  }

  /// @brief Returns true if the nodes are equal, excluding their children.
  friend bool
  topLevelEq(const std::shared_ptr<ReferenceNode> &lhs, const std::shared_ptr<ReferenceNode> &rhs) {
    return lhs->identifier == rhs->identifier && lhs->storedValue == rhs->storedValue &&
           lhs->lastWrite == rhs->lastWrite;
  }

  friend std::shared_ptr<ReferenceNode> greatestCommonSubtree(
      const std::shared_ptr<ReferenceNode> &lhs, const std::shared_ptr<ReferenceNode> &rhs
  ) {
    if (!topLevelEq(lhs, rhs)) {
      return nullptr;
    }
    auto res = lhs->clone(false); // childless clone
    // Find common children and recurse
    for (auto &[id, lhsChild] : lhs->children) {
      if (auto it = rhs->children.find(id); it != rhs->children.end()) {
        auto &rhsChild = it->second;
        if (auto gcs = greatestCommonSubtree(lhsChild, rhsChild)) {
          res->children[id] = gcs;
          if (id.isValue()) {
            ++res->dynamicChildCount;
          }
        }
      }
    }
    return res;
  }

private:
  ReferenceID identifier;
  mlir::Value storedValue;
  Operation *lastWrite;
  // Candidates are deliberately not copied into cloned states or value
  // snapshots. This keeps aggregate-read forwarding local to one allocation
  // path and avoids forwarding across a value-copy boundary.
  std::optional<Value> reusableAggregateRead;
  std::optional<AggregateSnapshotSource> aggregateSnapshotSource;
  std::optional<AggregateReadOrigin> aggregateReadOrigin;
  DenseMap<ReferenceID, std::shared_ptr<ReferenceNode>> children;
  // Number of direct dynamic children. Keep it synchronized with every children
  // mutation so constant-only invalidation remains independent of child fanout.
  size_t dynamicChildCount;
  uint64_t aggregateMutationEpoch = 0;

  template <typename IdType>
  ReferenceNode(IdType id, Value initialVal)
      : identifier(std::move(id)), storedValue(initialVal), lastWrite(nullptr), children(),
        dynamicChildCount(0) {}
};

using ValueMap = DenseMap<mlir::Value, std::shared_ptr<ReferenceNode>>;

/// Returns whether `type` denotes an aggregate that has value-copy rather than
/// SSA-alias semantics. `ReferenceNode::cloneForValueCopy` recursively copies
/// all nested aggregate state below such a root.
bool requiresAggregateSnapshot(Type type) {
  return isa<array::ArrayType, pod::PodType, component::StructType>(type);
}

/// The known contents of a global. Scalars can be forwarded by SSA identity,
/// while aggregates must retain an independent tree snapshot.
struct GlobalState {
  std::optional<Value> scalar;
  std::shared_ptr<ReferenceNode> aggregate;
};

using GlobalStateMap = DenseMap<SymbolRefAttr, GlobalState>;

/// Known values at a program point, split between tree-shaped value state and
/// flat stateful global/RAM facts.
struct KnownState {
  ValueMap values;
  GlobalStateMap globals;
  DenseMap<ReferenceID, Value> ram;
  // Unlike `ram`, only exact translated address values justify store removal.
  DenseMap<Value, Value> ramExact;
};

/// Writes eligible for removal only while traversing their containing block.
/// Candidates never cross CFG or nested-region boundaries.
struct BlockWriteCandidates {
  DenseMap<SymbolRefAttr, Operation *> globals;
  DenseMap<Value, Operation *> ram;

  void clear() {
    globals.clear();
    ram.clear();
  }
};

/// Intersects tree-shaped value facts by retaining only common subtrees.
ValueMap intersectValueMap(const ValueMap &lhs, const ValueMap &rhs) {
  ValueMap res;
  for (const auto &[id, lhsValTree] : lhs) {
    if (!lhsValTree) {
      continue;
    }
    if (auto it = rhs.find(id); it != rhs.end() && it->second) {
      // A missing common subtree is the conservative no-common-fact state.
      if (auto common = greatestCommonSubtree(lhsValTree, it->second)) {
        res[id] = std::move(common);
      }
    }
  }
  return res;
}

/// Intersect global facts conservatively. Scalar values retain the existing
/// forwarding behavior. Aggregate snapshots are dropped at joins rather than
/// being compared by allocation identity.
GlobalStateMap intersectGlobals(const GlobalStateMap &lhs, const GlobalStateMap &rhs) {
  GlobalStateMap res;
  for (const auto &[id, lhsState] : lhs) {
    if (!lhsState.scalar) {
      continue;
    }
    if (auto it = rhs.find(id); it != rhs.end() && it->second.scalar == lhsState.scalar) {
      res[id].scalar = lhsState.scalar;
    }
  }
  return res;
}

/// Intersects flat lookup facts by retaining keys mapped to the same value.
template <typename KeyT>
DenseMap<KeyT, Value>
intersectValueLookup(const DenseMap<KeyT, Value> &lhs, const DenseMap<KeyT, Value> &rhs) {
  DenseMap<KeyT, Value> res;
  for (const auto &[id, lhsVal] : lhs) {
    if (auto it = rhs.find(id); it != rhs.end() && it->second == lhsVal) {
      res[id] = lhsVal;
    }
  }
  return res;
}

/// Intersects known state across predecessor blocks.
KnownState intersect(const KnownState &lhs, const KnownState &rhs) {
  return {
      intersectValueMap(lhs.values, rhs.values), intersectGlobals(lhs.globals, rhs.globals),
      intersectValueLookup(lhs.ram, rhs.ram), intersectValueLookup(lhs.ramExact, rhs.ramExact)
  };
}

/// @brief Deep copy the ValueMap for when exclusive branches/regions need state
/// tracking, so that the orig state is not polluted through pointer updates.
ValueMap cloneValueMap(const ValueMap &orig) {
  ValueMap res;
  for (const auto &[id, tree] : orig) {
    res[id] = tree->clone();
  }
  return res;
}

GlobalStateMap cloneGlobalStateMap(const GlobalStateMap &orig) {
  GlobalStateMap res;
  for (const auto &[name, global] : orig) {
    res[name].scalar = global.scalar;
    if (global.aggregate) {
      res[name].aggregate = global.aggregate->clone();
    }
  }
  return res;
}

/// Deep copy the KnownState for exclusive branches/regions so tree updates do
/// not mutate the incoming state.
KnownState cloneKnownState(const KnownState &orig) {
  return {cloneValueMap(orig.values), cloneGlobalStateMap(orig.globals), orig.ram, orig.ramExact};
}

class PassImpl : public llzk::impl::RedundantReadAndWriteEliminationPassBase<PassImpl> {
  using Base = RedundantReadAndWriteEliminationPassBase<PassImpl>;
  using Base::Base;

  /// Aggregate values that are direct targets of a mutating operation anywhere
  /// in the current function. Such values cannot be safely merged with a
  /// distinct aggregate copy, even when their source access is unchanged.
  DenseSet<Value> aggregateWriteTargets;

  /// @brief Run the pass over the LLZK module. Currently the pass is intraprocedural,
  /// so this defers the optimization to `runOnFunc` for each function in the module.
  /// @note Due to MLIR limitations, you need to write passes as passes over ModuleOp,
  /// as setting them up as passes over FuncDefOp doesn't properly search all FuncDefOp
  /// and ultimately the pass does not run.
  void runOnOperation() override {
    getOperation().walk([this](FuncDefOp fn) { runOnFunc(fn); });
  }

  /// @brief Remove redundant reads and writes from the given function operation.
  /// @param fn
  void runOnFunc(FuncDefOp fn) {
    // Nothing to do for body-less functions.
    if (fn.getCallableRegion() == nullptr) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Running on " << fn.getName() << '\n');

    aggregateWriteTargets.clear();
    auto recordAggregateWriteTarget = [this](Value target) {
      if (requiresAggregateSnapshot(target.getType())) {
        aggregateWriteTargets.insert(target);
      }
    };
    fn.walk([&](Operation *op) {
      bool knownAggregateWrite = false;
      if (auto memberWrite = dyn_cast<MemberWriteOp>(op)) {
        recordAggregateWriteTarget(memberWrite.getComponent());
        knownAggregateWrite = true;
      } else if (auto arrayAccess = llvm::dyn_cast<ArrayAccessOpInterface>(op);
                 arrayAccess && !arrayAccess.isRead()) {
        recordAggregateWriteTarget(arrayAccess.getArrRef());
        knownAggregateWrite = true;
      } else if (auto podWrite = dyn_cast<pod::WritePodOp>(op)) {
        recordAggregateWriteTarget(podWrite.getPodRef());
        knownAggregateWrite = true;
      } else if (isa<global::GlobalWriteOp, ram::StoreOp, CallOp>(op)) {
        // These operations copy their source operand but do not mutate it.
        knownAggregateWrite = true;
      }
      if (!knownAggregateWrite && hasUnknownOrNonReadEffect(op)) {
        // An unmodeled effect may mutate any aggregate operand. Treating it
        // as a write target prevents a later RAUW from merging two copies.
        for (Value operand : op->getOperands()) {
          recordAggregateWriteTarget(operand);
        }
      }
    });

    // Maps redundant value -> necessary value.
    DenseMap<Value, Value> replacementMap;
    // All values created by a new_* operation or from a read*/extract* operation.
    SmallVector<Value> readVals;
    // All writes that are either (1) overwritten by subsequent writes or (2)
    // write a value that is already written.
    SmallVector<Operation *> redundantWrites;

    KnownState initState;
    // Initialize the state to the function arguments.
    for (auto arg : fn.getArguments()) {
      initState.values[arg] = ReferenceNode::create(arg, arg);
    }
    // Functions only have a single region
    (void)runOnRegion(
        *fn.getCallableRegion(), std::move(initState), replacementMap, readVals, redundantWrites
    );

    // Now that we have accumulated all necessary state, we perform the optimizations:
    // - Replace all redundant values.
    for (auto &[orig, replace] : replacementMap) {
      LLVM_DEBUG(llvm::dbgs() << "replacing " << orig << " with " << replace << '\n');
      orig.replaceAllUsesWith(replace);
      // We save the deletion to the readVals loop to prevent double-free.
    }
    // -Remove redundant writes now that it is safe to do so.
    for (auto *writeOp : redundantWrites) {
      LLVM_DEBUG(llvm::dbgs() << "erase write: " << *writeOp << '\n');
      writeOp->erase();
    }
    // - Now we do a pass over read values to see if any are now unused.
    // We do this in reverse order to free up early reads if their users would
    // be removed.
    for (auto it = readVals.rbegin(); it != readVals.rend(); it++) {
      Value readVal = *it;
      if (readVal.use_empty()) {
        LLVM_DEBUG(llvm::dbgs() << "erase read: " << readVal << '\n');
        readVal.getDefiningOp()->erase();
      }
    }
  }

  KnownState runOnRegion(
      Region &r, KnownState &&initState, DenseMap<Value, Value> &replacementMap,
      SmallVector<Value> &readVals, SmallVector<Operation *> &redundantWrites
  ) {
    // maps block -> state at the end of the block
    DenseMap<Block *, KnownState> endStates;
    // The first block has no predecessors, so nullptr contains the init state
    endStates[nullptr] = initState;
    auto getBlockState = [&endStates](Block *blockPtr) {
      auto it = endStates.find(blockPtr);
      ensure(it != endStates.end(), "unknown end state means we have an unsupported backedge");
      return cloneKnownState(it->second);
    };
    auto hasBlockState = [&endStates](Block *blockPtr) {
      return endStates.find(blockPtr) != endStates.end();
    };
    std::deque<Block *> frontier;
    DenseSet<Block *> queued;
    DenseSet<Block *> processed;
    auto enqueue = [&](Block *blockPtr) {
      if (processed.find(blockPtr) == processed.end() && queued.insert(blockPtr).second) {
        frontier.push_back(blockPtr);
      }
    };
    enqueue(&r.front());

    SmallVector<KnownState> terminalStates;
    size_t deferralsWithoutProgress = 0;

    while (!frontier.empty()) {
      Block *currentBlock = frontier.front();
      frontier.pop_front();
      queued.erase(currentBlock);

      // get predecessors
      KnownState currentState;
      auto it = currentBlock->pred_begin();
      auto itEnd = currentBlock->pred_end();
      if (it == itEnd) {
        // get the state for the entry block.
        currentState = getBlockState(nullptr);
      } else {
        bool ready = true;
        for (auto predIt = it; predIt != itEnd; predIt++) {
          ready &= hasBlockState(*predIt);
        }
        if (!ready) {
          deferralsWithoutProgress++;
          ensure(
              deferralsWithoutProgress <= frontier.size(),
              "unknown end state means we have an unsupported backedge"
          );
          enqueue(currentBlock);
          continue;
        }

        currentState = getBlockState(*it);
        // If we have multiple predecessors, we take a pessimistic view and
        // set the state as only the intersection of all predecessor states
        // (e.g., only the common state from an if branch).
        for (it++; it != itEnd; it++) {
          currentState = intersect(currentState, getBlockState(*it));
        }
      }

      // Run this block, consuming currentState and producing the endState
      deferralsWithoutProgress = 0;
      auto endState = runOnBlock(
          *currentBlock, std::move(currentState), replacementMap, readVals, redundantWrites
      );

      // Update the end states.
      // Since we only support the scf dialect, we should never have any
      // backedges, so we should never already have state for this block.
      ensure(processed.find(currentBlock) == processed.end(), "backedge");
      endStates[currentBlock] = std::move(endState);
      processed.insert(currentBlock);

      // add successors to frontier
      if (currentBlock->hasNoSuccessors()) {
        terminalStates.push_back(cloneKnownState(endStates[currentBlock]));
      } else {
        for (Block *succ : currentBlock->getSuccessors()) {
          enqueue(succ);
        }
      }
    }

    // The final state is the intersection of all possible terminal states.
    ensure(!terminalStates.empty(), "computed no states");
    auto finalState = terminalStates.front();
    for (const auto *it = terminalStates.begin() + 1; it != terminalStates.end(); it++) {
      finalState = intersect(finalState, *it);
    }
    return finalState;
  }

  KnownState runOnBlock(
      Block &b, KnownState &&state, DenseMap<Value, Value> &replacementMap,
      SmallVector<Value> &readVals, SmallVector<Operation *> &redundantWrites
  ) {
    BlockWriteCandidates writeCandidates;
    auto clearTreeWriteCandidates = [](KnownState &knownState) {
      for (auto &[_, valueTree] : knownState.values) {
        if (valueTree) {
          valueTree->clearLastWritesInSubtree();
        }
      }
    };

    for (Operation &op : b) {
      // Some operations have regions (e.g., scf.if). These regions must be
      // traversed and the resulting state(s) are intersected for the final
      // state of this operation.
      if (!op.getRegions().empty()) {
        KnownState parentState = cloneKnownState(state);
        // Repeating regions (scf.for, scf.while) execute their body more than
        // once. Pre-loop facts must not be used to declare a read inside the
        // body redundant — the body may observe writes from a previous
        // iteration.
        KnownState regionEntryState = cloneKnownState(state);
        // Tree-shaped reference last-write pointers are block-local deletion
        // candidates. Do not let an exclusive region inherit a candidate from
        // its parent, where one arm could otherwise queue it for function-wide
        // erasure.
        clearTreeWriteCandidates(regionEntryState);
        if (isa<scf::ForOp, scf::WhileOp>(op)) {
          regionEntryState.values.clear();
          regionEntryState.globals.clear();
          regionEntryState.ram.clear();
          regionEntryState.ramExact.clear();
        }
        SmallVector<KnownState> regionStates;
        for (Region &region : op.getRegions()) {
          if (region.empty()) {
            continue;
          }
          auto regionState = runOnRegion(
              region, cloneKnownState(regionEntryState), replacementMap, readVals, redundantWrites
          );
          regionStates.push_back(regionState);
        }
        if (regionStates.empty()) {
          // Region-bearing ops with no bodies still need their own effects handled.
          runOperation(&op, state, replacementMap, readVals, redundantWrites, writeCandidates);
          writeCandidates.clear();
          continue;
        }

        KnownState finalState = regionStates.front();
        for (const auto *it = regionStates.begin() + 1; it != regionStates.end(); it++) {
          finalState = intersect(finalState, *it);
        }
        // A nested region may be conditional, zero-iteration, or otherwise not
        // execute exactly once. Only propagate facts that remain true both
        // before and after the region traversal. In particular, a one-armed
        // scf.if must not make a write in its then-region appear unconditional.
        finalState.values = intersectValueMap(parentState.values, finalState.values);
        finalState.globals = intersectGlobals(parentState.globals, finalState.globals);
        finalState.ram = intersectValueLookup(parentState.ram, finalState.ram);
        finalState.ramExact = intersectValueLookup(parentState.ramExact, finalState.ramExact);
        // Likewise, do not export a tree-shaped reference candidate from one
        // region through the join. A later write must not treat an
        // exclusive-arm write as a block-local predecessor.
        clearTreeWriteCandidates(finalState);
        state = std::move(finalState);
        writeCandidates.clear();
        continue;
      }
      runOperation(&op, state, replacementMap, readVals, redundantWrites, writeCandidates);
    }
    return std::move(state);
  }

  /// @brief Perform the read/write operation contained in `op`, or do nothing
  /// if `op` is not a type of read/write operation.
  /// @param op An operation found in a LLZK function
  /// @param state Mutable state that is updated by executing `op`
  /// @param replacementMap A mutable map of original -> replacement values
  /// @param readVals A mutable list of all read values
  /// @param redundantWrites A mutable list of all writes that are considered redundant
  void runOperation(
      Operation *op, KnownState &state, DenseMap<Value, Value> &replacementMap,
      SmallVector<Value> &readVals, SmallVector<Operation *> &redundantWrites,
      BlockWriteCandidates &writeCandidates
  ) {
    // Uses the replacement map to look up values to simplify later replacement.
    // This avoids having a daisy chain of "replace B with A", "replace C with B",
    // etc.
    auto translate = [&replacementMap](Value v) {
      if (auto it = replacementMap.find(v); it != replacementMap.end()) {
        return it->second;
      }
      return v;
    };

    // Lookup the value tree in the current state or return nullptr.
    auto tryGetValTree = [&state](Value v) -> std::shared_ptr<ReferenceNode> {
      if (auto it = state.values.find(v); it != state.values.end()) {
        return it->second;
      }
      return nullptr;
    };

    auto doStatefulRead =
        [&]<typename KeyT>(Value resVal, DenseMap<KeyT, Value> &knownValues, const KeyT &key) {
      readVals.push_back(resVal);
      if (auto it = knownValues.find(key); it != knownValues.end()) {
        replacementMap[resVal] = it->second;
        return true;
      } else {
        knownValues[key] = resVal;
        state.values[resVal] = ReferenceNode::create(resVal, resVal);
        return false;
      }
    };

    auto copiedValueTree = [&](Value value) {
      if (auto tree = tryGetValTree(value); tree && requiresAggregateSnapshot(value.getType())) {
        auto copy = tree->cloneForValueCopy(value);
        // Copying observes all known source contents, so a later mutation of
        // the source cannot erase a write that initialized the copy.
        tree->clearLastWritesInSubtree();
        return copy;
      }
      return tryGetValTree(value);
    };

    auto readAggregateSnapshot = [](const std::shared_ptr<ReferenceNode> &tree, Value result,
                                    Operation *read) {
      auto copy = tree->cloneForValueCopy(result);
      copy->setAggregateReadOrigin(tree, read);
      // The read observes all contents of this allocation, so preserve writes
      // that initialized the independent value copy.
      tree->clearLastWritesInSubtree();
      return copy;
    };

    auto useMutatesAggregate = [](OpOperand &use) {
      Value value = use.get();
      Operation *user = use.getOwner();
      if (auto memberWrite = dyn_cast<MemberWriteOp>(user)) {
        return memberWrite.getComponent() == value;
      }
      if (auto arrayAccess = llvm::dyn_cast<ArrayAccessOpInterface>(user)) {
        return !arrayAccess.isRead() && arrayAccess.getArrRef() == value;
      }
      if (auto podWrite = dyn_cast<pod::WritePodOp>(user)) {
        return podWrite.getPodRef() == value;
      }
      if (isa<global::GlobalWriteOp, ram::StoreOp, CallOp>(user)) {
        // These operations consume a copied source value rather than mutate it.
        return false;
      }
      if (isa<constrain::ConstraintOpInterface>(user)) {
        // Constraints consume aggregate values but cannot mutate their storage.
        return false;
      }
      return hasUnknownOrNonReadEffect(user);
    };

    auto mayMutateAfter = [&](Value value, Operation *point) {
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (!useMutatesAggregate(use)) {
          continue;
        }
        // Candidates are not propagated across control-flow state clones, so
        // a write outside this block is conservatively treated as later.
        if (user->getBlock() != point->getBlock() || point->isBeforeInBlock(user)) {
          return true;
        }
      }
      return false;
    };

    auto mayMutateBetween = [&](Value value, Operation *before, Operation *after) {
      for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (!useMutatesAggregate(use)) {
          continue;
        }
        if (user->getBlock() != before->getBlock() || user->getBlock() != after->getBlock() ||
            (before->isBeforeInBlock(user) && user->isBeforeInBlock(after))) {
          return true;
        }
      }
      return false;
    };

    // A write records the aggregate source that was copied into an access. A
    // later aggregate read may reuse that source when it has not changed since
    // the copy and neither copy is mutated afterwards. The caller still
    // creates a snapshot first, so the read observes pending writes even when
    // its SSA result is subsequently replaced.
    auto tryForwardAggregateSnapshot = [&](const std::shared_ptr<ReferenceNode> &tree, Value result,
                                           Operation *read) -> std::shared_ptr<ReferenceNode> {
      const auto &snapshotSource = tree->getAggregateSnapshotSource();
      if (!snapshotSource || mayMutateBetween(snapshotSource->value, snapshotSource->write, read) ||
          mayMutateAfter(snapshotSource->value, read) || mayMutateAfter(result, read)) {
        return nullptr;
      }
      if (auto sourceTree = tryGetValTree(snapshotSource->value)) {
        replacementMap[result] = snapshotSource->value;
        return sourceTree;
      }
      return nullptr;
    };

    // Reuse a prior aggregate read only when replacing the later copy with the
    // earlier copy preserves value-copy semantics. In particular, neither
    // copy may be mutated, and the candidate must not have been mutated before
    // this read was reached.
    auto tryReuseAggregateRead = [&](const std::shared_ptr<ReferenceNode> &tree, Value result,
                                     Operation *read) -> std::shared_ptr<ReferenceNode> {
      const auto &candidate = tree->getReusableAggregateRead();
      if (!candidate) {
        return nullptr;
      }
      Operation *candidateRead = candidate->getDefiningOp();
      if (candidateRead == nullptr || mayMutateBetween(*candidate, candidateRead, read) ||
          mayMutateAfter(*candidate, read) || mayMutateAfter(result, read)) {
        return nullptr;
      }
      if (auto candidateTree = tryGetValTree(*candidate)) {
        replacementMap[result] = *candidate;
        return candidateTree;
      }
      return nullptr;
    };

    /// Apply the common snapshot/forward/reuse sequence for an aggregate read.
    ///
    /// `tryReuse` and `rememberRead` retain the few operation-specific reuse
    /// policies (member reads are deliberately more conservative), while all
    /// aggregate access kinds share the same snapshot and forwarding behavior.
    auto processAggregateRead = [&](const std::shared_ptr<ReferenceNode> &storage, Value result,
                                    Operation *read, auto &&tryReuse, auto &&rememberRead) {
      auto snapshot = readAggregateSnapshot(storage, result, read);
      if (auto sourceTree = tryForwardAggregateSnapshot(storage, result, read)) {
        return sourceTree;
      }
      if (auto reusableTree = tryReuse(storage, result, read)) {
        return reusableTree;
      }
      auto resultTree = std::move(snapshot);
      rememberRead(storage, result);
      return resultTree;
    };

    auto tryStandardAggregateReuse = [&](const std::shared_ptr<ReferenceNode> &storage,
                                         Value result, Operation *read) {
      return tryReuseAggregateRead(storage, result, read);
    };
    auto rememberAggregateRead = [](const std::shared_ptr<ReferenceNode> &storage, Value result) {
      storage->setReusableAggregateRead(result);
    };

    // Struct member reads use a function-wide mutation pre-scan as their
    // reuse policy. Keep that policy separate from the generic sequence above.
    auto tryMemberAggregateReuse = [&](const std::shared_ptr<ReferenceNode> &storage, Value result,
                                       Operation *) -> std::shared_ptr<ReferenceNode> {
      const auto &candidate = storage->getReusableAggregateRead();
      if (!candidate || aggregateWriteTargets.contains(*candidate) ||
          aggregateWriteTargets.contains(result)) {
        return nullptr;
      }
      if (auto candidateTree = tryGetValTree(*candidate)) {
        replacementMap[result] = *candidate;
        return candidateTree;
      }
      return nullptr;
    };
    auto rememberMemberAggregateRead = [&](const std::shared_ptr<ReferenceNode> &storage,
                                           Value result) {
      if (!aggregateWriteTargets.contains(result)) {
        storage->setReusableAggregateRead(result);
      }
    };

    // An aggregate write is redundant when it copies the same source snapshot
    // that is already stored at the destination and that source has not been
    // mutated since the earlier copy. The destination's state records writes
    // by exact access path, so any intervening potentially-aliasing write has
    // already invalidated or replaced this candidate.
    auto isRedundantAggregateWrite =
        [&](const std::optional<ReferenceNode::AggregateSnapshotSource> &previous, Value value,
            Operation *write) {
      return previous && previous->value == value &&
             !mayMutateBetween(value, previous->write, write);
    };

    auto isRedundantTreeWrite =
        [&](const std::shared_ptr<ReferenceNode> &destination,
            const std::optional<ReferenceNode::AggregateSnapshotSource> &previous, Value value,
            Operation *write) {
      if (!requiresAggregateSnapshot(value.getType())) {
        return destination->getStoredValue() == value;
      }
      return isRedundantAggregateWrite(previous, value, write);
    };

    /// Commit the shared tree bookkeeping for a non-redundant array, POD, or
    /// struct access write. Each caller supplies the ancestors whose aggregate
    /// snapshots were mutated by its particular access path.
    auto commitTreeWrite = [&](const std::shared_ptr<ReferenceNode> &destination, Value value,
                               Operation *write, const std::shared_ptr<ReferenceNode> &valueTree,
                               ArrayRef<std::shared_ptr<ReferenceNode>> mutatedNodes) {
      for (const auto &node : mutatedNodes) {
        node->markAggregateMutation();
      }
      destination->clearAggregateSnapshotSource();
      if (Operation *lastWrite = destination->updateLastWrite(write)) {
        LLVM_DEBUG(
            llvm::dbgs() << write->getName().getStringRef() << ": replacing " << lastWrite
                         << " with prior write " << *lastWrite << '\n'
        );
        redundantWrites.push_back(lastWrite);
      }
      destination->setCurrentValue(value, valueTree);
      if (requiresAggregateSnapshot(value.getType())) {
        destination->setAggregateSnapshotSource(value, write);
      }
    };

    // Returns whether an operation between two aggregate accesses has an
    // unmodeled effect. Known writes are checked through per-access mutation
    // epochs instead, which lets a write to an unrelated array element or
    // record remain transparent to a read-back write.
    auto hasUnknownEffectBetween = [](Operation *before, Operation *after) {
      if (before == nullptr || after == nullptr || before->getBlock() != after->getBlock() ||
          !before->isBeforeInBlock(after)) {
        return true;
      }
      for (Operation *n = before->getNextNode(); n != after; n = n->getNextNode()) {
        if (isa<MemberWriteOp, pod::WritePodOp, global::GlobalWriteOp, ram::StoreOp, CallOp,
                constrain::ConstraintOpInterface>(n) ||
            llvm::dyn_cast<ArrayAccessOpInterface>(n)) {
          continue;
        }
        if (hasUnknownOrNonReadEffect(n)) {
          return true;
        }
      }
      return false;
    };

    // A write that restores an unchanged snapshot to the same storage path is
    // redundant. The source snapshot remembers both its read location and the
    // location's mutation epoch. Writes on the same path, nested below it, or
    // through an alias advance that epoch; unrelated sibling writes do not.
    auto isRedundantAggregateReadBackWrite = [&](const std::shared_ptr<ReferenceNode> &destination,
                                                 Value source, Operation *write) {
      auto sourceTree = tryGetValTree(source);
      if (sourceTree == nullptr) {
        return false;
      }
      const auto &origin = sourceTree->getAggregateReadOrigin();
      return origin && origin->storage == destination &&
             origin->mutationEpoch == destination->getAggregateMutationEpoch() &&
             !mayMutateBetween(source, origin->read, write) &&
             !hasUnknownEffectBetween(origin->read, write);
    };

    // An omitted table offset denotes the current row.
    const IntegerAttr zeroTableOffset = IntegerAttr::get(IndexType::get(op->getContext()), 0);
    auto getMemberNode = [&](Value component, FlatSymbolRefAttr member) {
      std::shared_ptr<ReferenceNode> componentNode = tryGetValTree(translate(component));
      if (componentNode == nullptr) {
        return std::shared_ptr<ReferenceNode>();
      }
      return componentNode->getOrCreateChild(member);
    };

    auto getMemberAccessNode = [&](MemberReadOp readm) {
      std::shared_ptr<ReferenceNode> access =
          getMemberNode(readm.getComponent(), readm.getMemberNameAttr());
      if (access == nullptr) {
        return access;
      }
      access = access->getOrCreateChild(readm.getTableOffset().value_or(zeroTableOffset));
      if (!readm.getMapOperands().empty()) {
        access = access->getOrCreateChild(readm.getMapOpGroupSizesAttr());
        access = access->getOrCreateChild(readm.getNumDimsPerMapAttr());
      }
      for (auto mapOperands : readm.getMapOperands()) {
        for (Value operand : mapOperands) {
          access = access->getOrCreateChild(translate(operand));
        }
      }
      return access;
    };

    // Read a value from an array. This works on both readarr operations (which
    // return a scalar value) and extractarr operations (which return a subarray).
    auto doArrayReadLike = [&]<HasInterface<ArrayAccessOpInterface> OpClass>(OpClass readarr) {
      Value resVal = readarr.getResult();
      std::shared_ptr<ReferenceNode> currValTree = tryGetValTree(translate(readarr.getArrRef()));
      if (currValTree == nullptr) {
        state.values[resVal] = ReferenceNode::create(resVal, resVal);
        readVals.push_back(resVal);
        return;
      }

      std::shared_ptr<ReferenceNode> rootValTree = currValTree;
      SmallVector<ReferenceID> indices;
      bool hasDynamicIndex = false;
      for (Value origIdx : readarr.getIndices()) {
        Value idxVal = translate(origIdx);
        ReferenceID indexId(idxVal);
        hasDynamicIndex |= !indexId.isConst();
        indices.push_back(indexId);
        currValTree = currValTree->getOrCreateChild(idxVal);
      }

      if (requiresAggregateSnapshot(resVal.getType())) {
        if (hasDynamicIndex) {
          rootValTree->clearLastWritesObservedBy(indices);
        }
        state.values[resVal] = processAggregateRead(
            currValTree, resVal, readarr.getOperation(), tryStandardAggregateReuse,
            rememberAggregateRead
        );
        readVals.push_back(resVal);
        return;
      }

      if (!currValTree->hasStoredValue()) {
        currValTree->setCurrentValue(resVal);
      }

      if (currValTree->getStoredValue() != resVal) {
        LLVM_DEBUG(
            llvm::dbgs() << readarr.getOperationName() << ": replace " << resVal << " with "
                         << currValTree->getStoredValue() << '\n'
        );
        replacementMap[resVal] = currValTree->getStoredValue();
      } else {
        if (hasDynamicIndex) {
          rootValTree->clearLastWritesObservedBy(indices);
        }
        state.values[resVal] = currValTree;
        LLVM_DEBUG(
            llvm::dbgs() << readarr.getOperationName() << ": " << resVal << " => " << *currValTree
                         << '\n'
        );
      }

      readVals.push_back(resVal);
    };

    // Handle array.write (scalar) and array.insert (subarray) with one or more
    // indices. Dynamic indices may alias any sibling; constant indices only
    // invalidate dynamic-index siblings.
    auto doArrayWriteLike = [&]<HasInterface<ArrayAccessOpInterface> OpClass>(OpClass writearr) {
      std::shared_ptr<ReferenceNode> currValTree = tryGetValTree(translate(writearr.getArrRef()));
      if (currValTree == nullptr) {
        return;
      }
      Value newVal = translate(writearr.getRvalue());

      // Look up the destination without changing the access tree. This gives a
      // read-back write a chance to prove itself redundant before normal write
      // processing invalidates dynamic aliases or snapshot provenance.
      if (requiresAggregateSnapshot(newVal.getType())) {
        auto destination = currValTree;
        for (Value origIdx : writearr.getIndices()) {
          destination = destination->getChild(translate(origIdx));
          if (destination == nullptr) {
            break;
          }
        }
        if (destination != nullptr &&
            isRedundantAggregateReadBackWrite(destination, newVal, writearr.getOperation())) {
          redundantWrites.push_back(writearr.getOperation());
          return;
        }
      }

      std::shared_ptr<ReferenceNode> valTree = copiedValueTree(newVal);
      SmallVector<std::shared_ptr<ReferenceNode>> mutatedNodes = {currValTree};

      for (Value origIdx : writearr.getIndices()) {
        currValTree->clearAggregateSnapshotSource();
        Value idxVal = translate(origIdx);
        // A dynamic index may alias any sibling. A constant index only aliases
        // a dynamic sibling, so preserve unrelated constant-index facts.
        if (ReferenceID(idxVal).isConst()) {
          currValTree->invalidateDynamicChildren();
        } else {
          LLVM_DEBUG(llvm::dbgs() << writearr.getOperationName() << ": invalidate alias\n");
          currValTree->invalidateChildren();
        }
        currValTree = currValTree->getOrCreateChild(idxVal);
        mutatedNodes.push_back(currValTree);
      }
      std::optional<ReferenceNode::AggregateSnapshotSource> previousSnapshot =
          currValTree->getAggregateSnapshotSource();

      // A subarray write copies its rvalue. The same aggregate SSA value can
      // therefore denote a different snapshot after the source is mutated.
      // SSA identity is only sufficient to prove a scalar write redundant.
      if (isRedundantTreeWrite(currValTree, previousSnapshot, newVal, writearr.getOperation())) {
        LLVM_DEBUG(
            llvm::dbgs() << writearr.getOperationName() << ": subsequent " << writearr
                         << " is redundant\n"
        );
        redundantWrites.push_back(writearr);
      } else {
        // A write below an aggregate path invalidates read-back candidates for
        // that path and all of its ancestors, but not constant-index siblings.
        commitTreeWrite(currValTree, newVal, writearr.getOperation(), valTree, mutatedNodes);
      }
    };

    // global ops
    if (auto readGlobal = dyn_cast<global::GlobalReadOp>(op)) {
      const auto name = readGlobal.getNameRef();
      Value result = readGlobal.getVal();
      readVals.push_back(result);
      if (requiresAggregateSnapshot(result.getType())) {
        if (auto it = state.globals.find(name); it != state.globals.end() && it->second.aggregate) {
          state.values[result] = processAggregateRead(
              it->second.aggregate, result, op, tryStandardAggregateReuse, rememberAggregateRead
          );
        } else {
          auto globalState = ReferenceNode::create(result, result);
          state.values[result] = processAggregateRead(
              globalState, result, op, tryStandardAggregateReuse, rememberAggregateRead
          );
          state.globals[name] = GlobalState {
              .scalar = std::nullopt,
              .aggregate = std::move(globalState),
          };
        }
        writeCandidates.globals.erase(name);
      } else if (auto it = state.globals.find(name);
                 it != state.globals.end() && it->second.scalar) {
        replacementMap[result] = *it->second.scalar;
      } else {
        state.globals[name] = GlobalState {.scalar = result, .aggregate = nullptr};
        state.values[result] = ReferenceNode::create(result, result);
        writeCandidates.globals.erase(name);
      }
    } else if (auto writeGlobal = dyn_cast<global::GlobalWriteOp>(op)) {
      const auto name = writeGlobal.getNameRef();
      Value value = translate(writeGlobal.getVal());
      if (requiresAggregateSnapshot(value.getType()) && [&] {
        auto known = state.globals.find(name);
        return known != state.globals.end() && known->second.aggregate &&
               isRedundantAggregateReadBackWrite(
                   known->second.aggregate, value, writeGlobal.getOperation()
               );
      }()) {
        redundantWrites.push_back(writeGlobal.getOperation());
      } else if (!requiresAggregateSnapshot(value.getType()) && [&] {
        auto known = state.globals.find(name);
        return known != state.globals.end() && known->second.scalar == value;
      }()) {
        redundantWrites.push_back(writeGlobal.getOperation());
      } else if (requiresAggregateSnapshot(value.getType()) && [&] {
        auto known = state.globals.find(name);
        return known != state.globals.end() && known->second.aggregate &&
               isRedundantTreeWrite(
                   known->second.aggregate, known->second.aggregate->getAggregateSnapshotSource(),
                   value, writeGlobal.getOperation()
               );
      }()) {
        redundantWrites.push_back(writeGlobal.getOperation());
      } else {
        if (auto previous = writeCandidates.globals.find(name);
            previous != writeCandidates.globals.end()) {
          redundantWrites.push_back(previous->second);
        }
        if (requiresAggregateSnapshot(value.getType())) {
          if (auto valueTree = copiedValueTree(value)) {
            valueTree->setAggregateSnapshotSource(value, writeGlobal.getOperation());
            state.globals[name] = GlobalState {
                .scalar = std::nullopt,
                .aggregate = std::move(valueTree),
            };
          } else {
            state.globals.erase(name);
          }
        } else {
          state.globals[name] = GlobalState {.scalar = value, .aggregate = nullptr};
        }
        writeCandidates.globals[name] = writeGlobal.getOperation();
      }
    }
    // RAM ops
    else if (auto load = dyn_cast<ram::LoadOp>(op)) {
      Value address = translate(load.getAddr());
      if (!doStatefulRead(load.getVal(), state.ram, ReferenceID(address))) {
        writeCandidates.ram.clear();
      }
      state.ramExact[address] = translate(load.getVal());
    } else if (auto store = dyn_cast<ram::StoreOp>(op)) {
      Value address = translate(store.getAddr());
      Value value = translate(store.getVal());
      if (auto known = state.ramExact.find(address);
          known != state.ramExact.end() && known->second == value) {
        redundantWrites.push_back(store.getOperation());
      } else {
        if (auto previous = writeCandidates.ram.find(address);
            previous != writeCandidates.ram.end()) {
          redundantWrites.push_back(previous->second);
        }
        writeCandidates.ram[address] = store.getOperation();
        state.ram.clear();
        state.ramExact.clear();
        state.ram[ReferenceID(address)] = value;
        state.ramExact[address] = value;
      }
    }
    // struct ops
    else if (auto newStruct = dyn_cast<CreateStructOp>(op)) {
      // For new values, the "stored value" of the reference is the creation site.
      auto structVal = ReferenceNode::create(newStruct, newStruct);
      state.values[newStruct] = structVal;
      LLVM_DEBUG(
          llvm::dbgs() << newStruct.getOperationName() << ": " << *state.values[newStruct] << '\n'
      );
      // adding this to readVals
      readVals.push_back(newStruct);
    } else if (auto readm = dyn_cast<MemberReadOp>(op)) {
      std::shared_ptr<ReferenceNode> access = getMemberAccessNode(readm);
      Value resVal = readm.getVal();
      if (access == nullptr) {
        state.values[resVal] = ReferenceNode::create(resVal, resVal);
        readVals.push_back(resVal);
        return;
      }
      if (requiresAggregateSnapshot(resVal.getType())) {
        state.values[resVal] = processAggregateRead(
            access, resVal, readm.getOperation(), tryMemberAggregateReuse,
            rememberMemberAggregateRead
        );
        readVals.push_back(resVal);
        return;
      }
      if (!access->hasStoredValue()) {
        access->setCurrentValue(resVal);
      }
      if (access->getStoredValue() != resVal) {
        LLVM_DEBUG(
            llvm::dbgs() << readm.getOperationName() << ": adding replacement map entry { "
                         << resVal << " => " << access->getStoredValue() << " }\n"
        );
        replacementMap[resVal] = access->getStoredValue();
      } else {
        state.values[resVal] = access;
        LLVM_DEBUG(llvm::dbgs() << readm.getOperationName() << ": " << *access << '\n');
      }
      readVals.push_back(resVal);
    } else if (auto writem = dyn_cast<MemberWriteOp>(op)) {
      auto componentTree = tryGetValTree(translate(writem.getComponent()));
      std::shared_ptr<ReferenceNode> member =
          getMemberNode(writem.getComponent(), writem.getMemberNameAttr());
      if (member == nullptr) {
        return;
      }
      Value writeVal = translate(writem.getVal());
      auto access = member->getOrCreateChild(zeroTableOffset);
      std::optional<ReferenceNode::AggregateSnapshotSource> previousSnapshot =
          access->getAggregateSnapshotSource();
      if (requiresAggregateSnapshot(writeVal.getType()) &&
          isRedundantAggregateReadBackWrite(access, writeVal, writem.getOperation())) {
        redundantWrites.push_back(writem);
        return;
      }
      if (requiresAggregateSnapshot(writeVal.getType()) &&
          isRedundantTreeWrite(access, previousSnapshot, writeVal, writem.getOperation())) {
        redundantWrites.push_back(writem);
        return;
      }
      if (componentTree) {
        componentTree->clearAggregateSnapshotSource();
      }
      // Symbolic and affine offsets may resolve to the current row. Constant
      // nonzero offsets stay distinct from a current-row member write.
      bool invalidatedMayAliasRead = member->invalidateNonIntegerOffsetChildren();
      auto valTree = copiedValueTree(writeVal);

      if (invalidatedMayAliasRead) {
        access->clearLastWrite();
      }
      // Member writes copy aggregate values, so do not treat repeated source
      // SSA identity as repeated stored state.
      if (!requiresAggregateSnapshot(writeVal.getType()) &&
          isRedundantTreeWrite(access, previousSnapshot, writeVal, writem.getOperation())) {
        LLVM_DEBUG(
            llvm::dbgs() << writem.getOperationName() << ": recording redundant write " << writem
                         << '\n'
        );
        redundantWrites.push_back(writem);
      } else {
        // A member write changes the current-row access and the enclosing
        // member/component snapshots, but leaves other members independent.
        SmallVector<std::shared_ptr<ReferenceNode>> mutatedNodes;
        if (componentTree) {
          mutatedNodes.push_back(componentTree);
        }
        mutatedNodes.push_back(member);
        mutatedNodes.push_back(access);
        commitTreeWrite(access, writeVal, writem.getOperation(), valTree, mutatedNodes);
        LLVM_DEBUG(
            llvm::dbgs() << writem.getOperationName() << ": " << *access << " set to " << writeVal
                         << '\n'
        );
      }
    }
    // array ops
    else if (auto newArray = dyn_cast<CreateArrayOp>(op)) {
      auto arrayVal = ReferenceNode::create(newArray, newArray);
      state.values[newArray] = arrayVal;

      // If we're given a constructor, we can instantiate elements using
      // constant indices.
      unsigned idx = 0;
      for (auto elem : newArray.getElements()) {
        Value elemVal = translate(elem);
        auto valTree = copiedValueTree(elemVal);
        auto elemChild = arrayVal->createChild(idx, elemVal, valTree);
        LLVM_DEBUG(
            llvm::dbgs() << newArray.getOperationName() << ": element " << idx << " initialized to "
                         << *elemChild << '\n'
        );
        idx++;
      }

      readVals.push_back(newArray);
    } else if (auto readarr = dyn_cast<ReadArrayOp>(op)) {
      doArrayReadLike(readarr);
    } else if (auto writearr = dyn_cast<WriteArrayOp>(op)) {
      doArrayWriteLike(writearr);
    } else if (auto extractarr = dyn_cast<ExtractArrayOp>(op)) {
      // Logic is essentially the same as readarr
      doArrayReadLike(extractarr);
    } else if (auto insertarr = dyn_cast<InsertArrayOp>(op)) {
      // Logic is essentially the same as writearr
      doArrayWriteLike(insertarr);
    } else if (auto newPod = dyn_cast<pod::NewPodOp>(op)) {
      Value podValue = newPod.getResult();
      auto podTree = ReferenceNode::create(podValue, podValue);
      state.values[podValue] = podTree;
      for (const auto &record : newPod.getInitializedRecordValues()) {
        Value value = translate(record.value);
        podTree->createChild(
            StringAttr::get(op->getContext(), record.name), value, copiedValueTree(value)
        );
      }
      readVals.push_back(podValue);
    } else if (auto readPod = dyn_cast<pod::ReadPodOp>(op)) {
      Value result = readPod.getResult();
      auto podTree = tryGetValTree(translate(readPod.getPodRef()));
      if (podTree == nullptr) {
        state.values[result] = ReferenceNode::create(result, result);
      } else {
        auto record = podTree->getOrCreateChild(readPod.getRecordNameAttr());
        if (requiresAggregateSnapshot(result.getType())) {
          state.values[result] = processAggregateRead(
              record, result, readPod.getOperation(), tryStandardAggregateReuse,
              rememberAggregateRead
          );
        } else if (!record->hasStoredValue()) {
          record->setCurrentValue(result);
          state.values[result] = record;
        } else if (record->getStoredValue() != result) {
          replacementMap[result] = record->getStoredValue();
        } else {
          state.values[result] = record;
        }
      }
      readVals.push_back(result);
    } else if (auto writePod = dyn_cast<pod::WritePodOp>(op)) {
      auto podTree = tryGetValTree(translate(writePod.getPodRef()));
      if (podTree == nullptr) {
        return;
      }
      Value value = translate(writePod.getValue());
      auto record = podTree->getOrCreateChild(writePod.getRecordNameAttr());
      std::optional<ReferenceNode::AggregateSnapshotSource> previousSnapshot =
          record->getAggregateSnapshotSource();
      if (requiresAggregateSnapshot(value.getType()) &&
          isRedundantAggregateReadBackWrite(record, value, writePod.getOperation())) {
        redundantWrites.push_back(writePod.getOperation());
        return;
      }
      if (requiresAggregateSnapshot(value.getType()) &&
          isRedundantTreeWrite(record, previousSnapshot, value, writePod.getOperation())) {
        redundantWrites.push_back(writePod.getOperation());
        return;
      }
      podTree->clearAggregateSnapshotSource();
      // POD records also store aggregate snapshots rather than aliases.
      if (!requiresAggregateSnapshot(value.getType()) &&
          isRedundantTreeWrite(record, previousSnapshot, value, writePod.getOperation())) {
        redundantWrites.push_back(writePod.getOperation());
      } else {
        // Record writes affect this record and the enclosing POD, but not
        // unrelated records' copied aggregate snapshots.
        SmallVector<std::shared_ptr<ReferenceNode>> mutatedNodes = {podTree, record};
        commitTreeWrite(
            record, value, writePod.getOperation(), copiedValueTree(value), mutatedNodes
        );
      }
    } else if (hasUnknownOrNonReadEffect(op)) {
      // Calls and constraints consume aggregate operands by value. Other
      // unmodeled effects may mutate aggregate operands, so discard their
      // reference-tree state before a later read or write can reuse stale
      // contents. Other aggregate values remain valid because aggregate
      // copies have value-copy semantics.
      if (!isa<CallOp, constrain::ConstraintOpInterface>(op)) {
        for (Value operand : op->getOperands()) {
          if (requiresAggregateSnapshot(operand.getType())) {
            state.values.erase(operand);
            state.values.erase(translate(operand));
          }
        }
      }
      state.globals.clear();
      state.ram.clear();
      state.ramExact.clear();
      writeCandidates.clear();
    } else if (hasReadEffect(op)) {
      // A read does not invalidate known values, but it can observe a pending
      // write and therefore prevents removing that write as overwritten.
      writeCandidates.clear();
    }
  }
};

} // namespace
