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
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/Concepts.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

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

/// @brief An reference to a value, represented either by an SSA value, a
/// symbol reference (e.g., a member name), or an int (e.g., a constant array index).
class ReferenceID {
public:
  explicit ReferenceID(Value v) {
    // reserved special pointer values for DenseMapInfo
    if (v.getImpl() == reinterpret_cast<mlir::detail::ValueImpl *>(1) ||
        v.getImpl() == reinterpret_cast<mlir::detail::ValueImpl *>(2)) {
      identifier = v;
    } else if (auto constVal = dyn_cast_if_present<FeltConstantOp>(v.getDefiningOp())) {
      identifier = constVal.getValue();
    } else if (auto constIdxVal = dyn_cast_if_present<arith::ConstantIndexOp>(v.getDefiningOp())) {
      identifier = llvm::cast<IntegerAttr>(constIdxVal.getValue()).getValue();
    } else {
      identifier = v;
    }
  }
  explicit ReferenceID(FlatSymbolRefAttr s) : identifier(s) {}
  explicit ReferenceID(const APInt &i) : identifier(i) {}
  explicit ReferenceID(unsigned i) : identifier(APInt(64, i)) {}

  bool isValue() const { return std::holds_alternative<Value>(identifier); }
  bool isSymbol() const { return std::holds_alternative<FlatSymbolRefAttr>(identifier); }
  bool isConst() const { return std::holds_alternative<APInt>(identifier); }

  Value getValue() const {
    ensure(isValue(), "does not hold Value");
    return std::get<Value>(identifier);
  }

  FlatSymbolRefAttr getSymbol() const {
    ensure(isSymbol(), "does not hold symbol");
    return std::get<FlatSymbolRefAttr>(identifier);
  }

  APInt getConst() const {
    ensure(isConst(), "does not hold const");
    return std::get<APInt>(identifier);
  }

  void print(raw_ostream &os) const {
    if (auto v = std::get_if<Value>(&identifier)) {
      if (auto opres = dyn_cast<OpResult>(*v)) {
        os << '%' << opres.getResultNumber();
      } else {
        os << *v;
      }
    } else if (auto s = std::get_if<FlatSymbolRefAttr>(&identifier)) {
      os << *s;
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
  /// FlatSymbolRefAttr: identifier refers to a named member in a struct
  /// APInt: identifier refers to a constant index in an array
  /// Value: identifier refers to a dynamic index in an array
  std::variant<FlatSymbolRefAttr, APInt, Value> identifier;
};

} // namespace

namespace llvm {

/// @brief Allows ReferenceID to be a DenseMap key.
template <> struct DenseMapInfo<ReferenceID> {
  static ReferenceID getEmptyKey() {
    return ReferenceID(mlir::Value(reinterpret_cast<mlir::detail::ValueImpl *>(1)));
  }
  static inline ReferenceID getTombstoneKey() {
    return ReferenceID(mlir::Value(reinterpret_cast<mlir::detail::ValueImpl *>(2)));
  }
  static unsigned getHashValue(const ReferenceID &r) {
    if (r.isValue()) {
      return hash_value(r.getValue());
    } else if (r.isSymbol()) {
      return hash_value(r.getSymbol());
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
///
/// Values not in this tree are unknown, and therefore not subject to read/write
/// elimination until they become known and can be eliminated when redundant operations
/// are performed.
///
/// A node may have "constant identifiers" as children (member refs, constant indices)
/// or a single non-constant child index (just an mlir::Value), as the dynamic
/// index may or may not alias any constant identifiers. If a dynamic index is
/// added, the user should clear the prior known children to prevent accidental aliasing.
///
/// Does not allow mixing of constant and non-constant child indices, as we
/// do not know if they alias.
class ReferenceNode {
public:
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
    if (withChildren) {
      for (const auto &[id, child] : children) {
        copy.children[id] = child->clone(withChildren);
      }
    }
    return std::make_shared<ReferenceNode>(std::move(copy));
  }

  template <typename IdType>
  std::shared_ptr<ReferenceNode>
  createChild(IdType id, Value storedVal, std::shared_ptr<ReferenceNode> valTree = nullptr) {
    std::shared_ptr<ReferenceNode> child = create(id, storedVal);
    child->setCurrentValue(storedVal, valTree);
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

  void setCurrentValue(Value v, std::shared_ptr<ReferenceNode> valTree = nullptr) {
    storedValue = v;
    if (valTree != nullptr) {
      // Overwrite our current set of children with new children, since we overwrote
      // the stored value.
      children = valTree->children;
    }
  }

  void invalidateChildren() { children.clear(); }

  bool isLeaf() const { return children.empty(); }

  Value getStoredValue() const { return storedValue; }

  bool hasStoredValue() const { return storedValue != nullptr; }

  void print(raw_ostream &os, int indent = 0) const {
    os.indent(indent) << '[' << identifier;
    if (storedValue != nullptr) {
      os << " => " << storedValue;
    }
    os << ']';
    if (!children.empty()) {
      os << "{\n";
      for (auto &[_, child] : children) {
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
        }
      }
    }
    return res;
  }

private:
  ReferenceID identifier;
  mlir::Value storedValue;
  Operation *lastWrite;
  DenseMap<ReferenceID, std::shared_ptr<ReferenceNode>> children;

  template <typename IdType>
  ReferenceNode(IdType id, Value initialVal)
      : identifier(id), storedValue(initialVal), lastWrite(nullptr), children() {}
};

using ValueMap = DenseMap<mlir::Value, std::shared_ptr<ReferenceNode>>;

ValueMap intersect(const ValueMap &lhs, const ValueMap &rhs) {
  ValueMap res;
  for (auto &[id, lhsValTree] : lhs) {
    if (auto it = rhs.find(id); it != rhs.end()) {
      auto &rhsValTree = it->second;
      res[id] = greatestCommonSubtree(lhsValTree, rhsValTree);
    }
  }
  return res;
}

/// @brief Deep copy the ValueMap for when exclusive branches/regions need state
/// tracking, so that the orig state is not polluted through pointer updates.
ValueMap cloneValueMap(const ValueMap &orig) {
  ValueMap res;
  for (auto &[id, tree] : orig) {
    res[id] = tree->clone();
  }
  return res;
}

class RedundantReadAndWriteEliminationPass
    : public llzk::impl::RedundantReadAndWriteEliminationPassBase<
          RedundantReadAndWriteEliminationPass> {
  /// @brief Run the pass over the LLZK module. Currently the pass is intraprocedural,
  /// so this defers the optimization to `runOnFunc` for each function in the module.
  /// @note Due to MLIR limitations, you need to write passes as passes over ModuleOp,
  /// as setting them up as passes over FuncDefOp doesn't properly search all FuncDefOp
  /// and ultimately the pass does not run.
  void runOnOperation() override {
    getOperation().walk([&](FuncDefOp fn) { runOnFunc(fn); });
  }

  /// @brief Remove redundant reads and writes from the given function operation.
  /// @param fn
  void runOnFunc(FuncDefOp fn) {
    // Nothing to do for body-less functions.
    if (fn.getCallableRegion() == nullptr) {
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Running on " << fn.getName() << '\n');

    // Maps redundant value -> necessary value.
    DenseMap<Value, Value> replacementMap;
    // All values created by a new_* operation or from a read*/extract* operation.
    SmallVector<Value> readVals;
    // All writes that are either (1) overwritten by subsequent writes or (2)
    // write a value that is already written.
    SmallVector<Operation *> redundantWrites;

    ValueMap initState;
    // Initialize the state to the function arguments.
    for (auto arg : fn.getArguments()) {
      initState[arg] = ReferenceNode::create(arg, arg);
    }
    // Functions only have a single region
    (void)runOnRegion(
        *fn.getCallableRegion(), std::move(initState), replacementMap, readVals, redundantWrites
    );

    // Now that we have accumulated all necessary state, we perform the optimizations:
    // - Replace all redundant values.
    for (auto &[orig, replace] : replacementMap) {
      LLVM_DEBUG(llvm::dbgs() << "replacing " << orig << " with " << orig << '\n');
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

  ValueMap runOnRegion(
      Region &r, ValueMap &&initState, DenseMap<Value, Value> &replacementMap,
      SmallVector<Value> &readVals, SmallVector<Operation *> &redundantWrites
  ) {
    // maps block -> state at the end of the block
    DenseMap<Block *, ValueMap> endStates;
    // The first block has no predecessors, so nullptr contains the init state
    endStates[nullptr] = initState;
    auto getBlockState = [&endStates](Block *blockPtr) {
      auto it = endStates.find(blockPtr);
      ensure(it != endStates.end(), "unknown end state means we have an unsupported backedge");
      return cloneValueMap(it->second);
    };
    std::deque<Block *> frontier;
    frontier.push_back(&r.front());
    DenseSet<Block *> visited;

    SmallVector<std::reference_wrapper<const ValueMap>> terminalStates;

    while (!frontier.empty()) {
      Block *currentBlock = frontier.front();
      frontier.pop_front();
      visited.insert(currentBlock);

      // get predecessors
      ValueMap currentState;
      auto it = currentBlock->pred_begin();
      auto itEnd = currentBlock->pred_end();
      if (it == itEnd) {
        // get the state for the entry block.
        currentState = getBlockState(nullptr);
      } else {
        currentState = getBlockState(*it);
        // If we have multiple predecessors, we take a pessimistic view and
        // set the state as only the intersection of all predecessor states
        // (e.g., only the common state from an if branch).
        for (it++; it != itEnd; it++) {
          currentState = intersect(currentState, getBlockState(*it));
        }
      }

      // Run this block, consuming currentState and producing the endState
      auto endState = runOnBlock(
          *currentBlock, std::move(currentState), replacementMap, readVals, redundantWrites
      );

      // Update the end states.
      // Since we only support the scf dialect, we should never have any
      // backedges, so we should never already have state for this block.
      ensure(endStates.find(currentBlock) == endStates.end(), "backedge");
      endStates[currentBlock] = std::move(endState);

      // add successors to frontier
      if (currentBlock->hasNoSuccessors()) {
        terminalStates.push_back(endStates[currentBlock]);
      } else {
        for (Block *succ : currentBlock->getSuccessors()) {
          if (visited.find(succ) == visited.end()) {
            frontier.push_back(succ);
          }
        }
      }
    }

    // The final state is the intersection of all possible terminal states.
    ensure(!terminalStates.empty(), "computed no states");
    auto finalState = terminalStates.front().get();
    for (auto it = terminalStates.begin() + 1; it != terminalStates.end(); it++) {
      finalState = intersect(finalState, it->get());
    }
    return finalState;
  }

  ValueMap runOnBlock(
      Block &b, ValueMap &&state, DenseMap<Value, Value> &replacementMap,
      SmallVector<Value> &readVals, SmallVector<Operation *> &redundantWrites
  ) {
    for (Operation &op : b) {
      runOperation(&op, state, replacementMap, readVals, redundantWrites);
      // Some operations have regions (e.g., scf.if). These regions must be
      // traversed and the resulting state(s) are intersected for the final
      // state of this operation.
      if (!op.getRegions().empty()) {
        SmallVector<ValueMap> regionStates;
        for (Region &region : op.getRegions()) {
          auto regionState =
              runOnRegion(region, cloneValueMap(state), replacementMap, readVals, redundantWrites);
          regionStates.push_back(regionState);
        }

        ValueMap finalState = regionStates.front();
        for (auto it = regionStates.begin() + 1; it != regionStates.end(); it++) {
          finalState = intersect(finalState, *it);
        }
        state = std::move(finalState);
      }
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
      Operation *op, ValueMap &state, DenseMap<Value, Value> &replacementMap,
      SmallVector<Value> &readVals, SmallVector<Operation *> &redundantWrites
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
      if (auto it = state.find(v); it != state.end()) {
        return it->second;
      }
      return nullptr;
    };

    // Read a value from an array. This works on both readarr operations (which
    // return a scalar value) and extractarr operations (which return a subarray).
    auto doArrayReadLike = [&]<HasInterface<ArrayAccessOpInterface> OpClass>(OpClass readarr) {
      std::shared_ptr<ReferenceNode> currValTree = state.at(translate(readarr.getArrRef()));

      for (Value origIdx : readarr.getIndices()) {
        Value idxVal = translate(origIdx);
        currValTree = currValTree->getOrCreateChild(idxVal);
      }

      Value resVal = readarr.getResult();
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
        state[resVal] = currValTree;
        LLVM_DEBUG(
            llvm::dbgs() << readarr.getOperationName() << ": " << resVal << " => " << *currValTree
                         << '\n'
        );
      }

      readVals.push_back(resVal);
    };

    // Write a scalar value (for writearr) or a subarray value (for insertarr)
    // to an array. The unique part of this operation relative to others is that
    // we may receive a variable index (i.e., not a constant). In this case, we
    // invalidate ajoining parts of the subtree, since it is possible that
    // the variable index aliases one of the other elements and may or may not
    // override that value.
    auto doArrayWriteLike = [&]<HasInterface<ArrayAccessOpInterface> OpClass>(OpClass writearr) {
      std::shared_ptr<ReferenceNode> currValTree = state.at(translate(writearr.getArrRef()));
      Value newVal = translate(writearr.getRvalue());
      std::shared_ptr<ReferenceNode> valTree = tryGetValTree(newVal);

      for (Value origIdx : writearr.getIndices()) {
        Value idxVal = translate(origIdx);
        // This write will invalidate all children, since it may reference
        // any number of them.
        if (ReferenceID(idxVal).isValue()) {
          LLVM_DEBUG(llvm::dbgs() << writearr.getOperationName() << ": invalidate alias\n");
          currValTree->invalidateChildren();
        }
        currValTree = currValTree->getOrCreateChild(idxVal);
      }

      if (currValTree->getStoredValue() == newVal) {
        LLVM_DEBUG(
            llvm::dbgs() << writearr.getOperationName() << ": subsequent " << writearr
                         << " is redundant\n"
        );
        redundantWrites.push_back(writearr);
      } else {
        if (Operation *lastWrite = currValTree->updateLastWrite(writearr)) {
          LLVM_DEBUG(
              llvm::dbgs() << writearr.getOperationName() << "writearr: replacing " << lastWrite
                           << " with prior write " << *lastWrite << '\n'
          );
          redundantWrites.push_back(lastWrite);
        }
        currValTree->setCurrentValue(newVal, valTree);
      }
    };

    // struct ops
    if (auto newStruct = dyn_cast<CreateStructOp>(op)) {
      // For new values, the "stored value" of the reference is the creation site.
      auto structVal = ReferenceNode::create(newStruct, newStruct);
      state[newStruct] = structVal;
      LLVM_DEBUG(llvm::dbgs() << newStruct.getOperationName() << ": " << *state[newStruct] << '\n');
      // adding this to readVals
      readVals.push_back(newStruct);
    } else if (auto readm = dyn_cast<MemberReadOp>(op)) {
      auto structVal = state.at(translate(readm.getComponent()));
      FlatSymbolRefAttr symbol = readm.getMemberNameAttr();
      Value resVal = translate(readm.getVal());
      // Check if such a child already exists.
      if (auto child = structVal->getChild(symbol)) {
        LLVM_DEBUG(
            llvm::dbgs() << readm.getOperationName() << ": adding replacement map entry { "
                         << resVal << " => " << child->getStoredValue() << " }\n"
        );
        replacementMap[resVal] = child->getStoredValue();
      } else {
        // If we have no previous store, we create a new symbolic value for
        // this location.
        state[readm] = structVal->createChild(symbol, resVal);
        LLVM_DEBUG(llvm::dbgs() << readm.getOperationName() << ": " << *state[readm] << '\n');
      }
      // specifically add the untranslated value back for removal checks
      readVals.push_back(readm.getVal());
    } else if (auto writem = dyn_cast<MemberWriteOp>(op)) {
      auto structVal = state.at(translate(writem.getComponent()));
      Value writeVal = translate(writem.getVal());
      FlatSymbolRefAttr symbol = writem.getMemberNameAttr();
      auto valTree = tryGetValTree(writeVal);

      auto child = structVal->getOrCreateChild(symbol);
      if (child->getStoredValue() == writeVal) {
        LLVM_DEBUG(
            llvm::dbgs() << writem.getOperationName() << ": recording redundant write " << writem
                         << '\n'
        );
        redundantWrites.push_back(writem);
      } else {
        if (auto *lastWrite = child->updateLastWrite(writem)) {
          LLVM_DEBUG(
              llvm::dbgs() << writem.getOperationName() << ": recording overwritten write "
                           << *lastWrite << '\n'
          );
          redundantWrites.push_back(lastWrite);
        }
        child->setCurrentValue(writeVal, valTree);
        LLVM_DEBUG(
            llvm::dbgs() << writem.getOperationName() << ": " << *child << " set to " << writeVal
                         << '\n'
        );
      }
    }
    // array ops
    else if (auto newArray = dyn_cast<CreateArrayOp>(op)) {
      auto arrayVal = ReferenceNode::create(newArray, newArray);
      state[newArray] = arrayVal;

      // If we're given a constructor, we can instantiate elements using
      // constant indices.
      unsigned idx = 0;
      for (auto elem : newArray.getElements()) {
        Value elemVal = translate(elem);
        auto valTree = tryGetValTree(elemVal);
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
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createRedundantReadAndWriteEliminationPass() {
  return std::make_unique<RedundantReadAndWriteEliminationPass>();
};
