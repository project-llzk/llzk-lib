//===-- LLZKRedundantOperationEliminationPass.cpp ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-duplicate-op-elim` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/RAM/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>

#include <utility>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_REDUNDANTOPERATIONELIMINATIONPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::boolean;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::function;

#define DEBUG_TYPE "llzk-duplicate-op-elim"

namespace {

static Operation *EMPTY_OP_KEY = llvm::DenseMapInfo<Operation *>::getEmptyKey();
static Operation *TOMBSTONE_OP_KEY = llvm::DenseMapInfo<Operation *>::getTombstoneKey();

// Maps original -> replacement value
using TranslationMap = DenseMap<Value, Value>;

// Stateful reads are equivalent only when they observe the same mutation history.
struct ReadStateKey {
  Block *block = nullptr;
  Operation *lastCommonBarrier = nullptr;
  Operation *lastStateMutation = nullptr;

  friend bool operator==(const ReadStateKey &lhs, const ReadStateKey &rhs) {
    return lhs.block == rhs.block && lhs.lastCommonBarrier == rhs.lastCommonBarrier &&
           lhs.lastStateMutation == rhs.lastStateMutation;
  }
};

struct BlockReadState {
  // Unknown and generic non-read effects invalidate every modeled read.
  Operation *lastCommonBarrier = nullptr;
  Operation *lastRamStore = nullptr;
  DenseMap<SymbolRefAttr, Operation *> lastGlobalWrite;
};

static bool hasUnknownOrNonReadEffect(Operation *op) {
  auto effects = getEffectsRecursively(op);
  return !effects || llvm::any_of(*effects, [](const MemoryEffects::EffectInstance &effect) {
    return !isa<MemoryEffects::Read>(effect.getEffect());
  });
}

static bool isDuplicateEliminationCandidate(Operation *op) {
  if (isa<NonDetOp>(op) || op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() != 0 ||
      op->getNumSuccessors() != 0) {
    return false;
  }

  // Constraint elimination is an intentional behavior of this pass. Global
  // and RAM reads are handled using the state keys below.
  return isa<ConstraintOpInterface, global::GlobalReadOp, ram::LoadOp>(op) ||
         isMemoryEffectFree(op);
}

static ReadStateKey getReadStateKey(Operation *op, const BlockReadState &state) {
  if (auto read = dyn_cast<global::GlobalReadOp>(op)) {
    return {
        op->getBlock(), state.lastCommonBarrier, state.lastGlobalWrite.lookup(read.getNameRef())
    };
  }
  if (isa<ram::LoadOp>(op)) {
    return {op->getBlock(), state.lastCommonBarrier, state.lastRamStore};
  }
  return {};
}

static void updateReadState(Operation *op, BlockReadState &state) {
  if (auto write = dyn_cast<global::GlobalWriteOp>(op)) {
    state.lastGlobalWrite[write.getNameRef()] = op;
    return;
  }

  if (isa<ram::StoreOp>(op)) {
    // Without RAM alias analysis, every store may affect every load.
    state.lastRamStore = op;
    return;
  }

  if (isa<ConstraintOpInterface, global::GlobalReadOp, ram::LoadOp>(op) || isMemoryEffectFree(op)) {
    return;
  }

  if (hasUnknownOrNonReadEffect(op)) {
    state.lastCommonBarrier = op;
  }
}

static bool isDeadAfterElimination(Operation *op) {
  if (isOpTriviallyDead(op)) {
    return true;
  }

  // Member reads are observations with no mutation. Keep this local so the
  // pass can clean up reads made unused by its rewrites without changing
  // dialect-wide canonicalization behavior for unrelated pipelines.
  return isa<MemberReadOp>(op) &&
         llvm::all_of(op->getResults(), [](Value result) { return result.use_empty(); });
}

/// @brief A wrapper for an operation that provides comparators for operations
/// to determine if their outputs will be equal. In general, this will compare
/// to see if the translated operands for a given operation are equal.
class OperationComparator {
public:
  explicit OperationComparator(Operation *o) : op(o) {
    if (op != EMPTY_OP_KEY && op != TOMBSTONE_OP_KEY) {
      operands = SmallVector<Value>(op->getOperands());
    }
  }

  OperationComparator(Operation *o, const TranslationMap &m, ReadStateKey key = {})
      : op(o), readStateKey(key) {
    for (Value operand : op->getOperands()) {
      if (auto it = m.find(operand); it != m.end()) {
        operands.push_back(it->second);
      } else {
        operands.push_back(operand);
      }
    }
  }

  Operation *getOp() const { return op; }

  const SmallVector<Value> &getOperands() const { return operands; }
  const ReadStateKey &getReadStateKey() const { return readStateKey; }

  bool isCommutative() const { return op->hasTrait<OpTrait::IsCommutative>(); }

  friend bool operator==(const OperationComparator &lhs, const OperationComparator &rhs) {
    if (lhs.op == EMPTY_OP_KEY || rhs.op == EMPTY_OP_KEY || lhs.op == TOMBSTONE_OP_KEY ||
        rhs.op == TOMBSTONE_OP_KEY) {
      return lhs.op == rhs.op;
    }

    if (!(lhs.readStateKey == rhs.readStateKey) ||
        !OperationEquivalence::isEquivalentTo(
            lhs.op, rhs.op, OperationEquivalence::ignoreValueEquivalence,
            /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations
        )) {
      return false;
    }

    // Preserve the pass's existing commutative matching for binary operations.
    // For a future n-ary commutative op, exact operand order remains conservative.
    if (lhs.isCommutative() && lhs.operands.size() == 2) {
      return (lhs.operands[0] == rhs.operands[0] && lhs.operands[1] == rhs.operands[1]) ||
             (lhs.operands[0] == rhs.operands[1] && lhs.operands[1] == rhs.operands[0]);
    }

    return lhs.operands == rhs.operands;
  }

private:
  Operation *op;
  SmallVector<Value> operands;
  ReadStateKey readStateKey;
};

} // namespace

namespace llvm {

template <> struct DenseMapInfo<OperationComparator> {
  static OperationComparator getEmptyKey() { return OperationComparator(EMPTY_OP_KEY); }
  static inline OperationComparator getTombstoneKey() {
    return OperationComparator(TOMBSTONE_OP_KEY);
  }
  static unsigned getHashValue(const OperationComparator &oc) {
    if (oc.getOp() == EMPTY_OP_KEY || oc.getOp() == TOMBSTONE_OP_KEY) {
      return hash_value(oc.getOp());
    }

    hash_code opHash = mlir::OperationEquivalence::computeHash(
        oc.getOp(), mlir::OperationEquivalence::ignoreHashValue,
        mlir::OperationEquivalence::ignoreHashValue, mlir::OperationEquivalence::IgnoreLocations
    );

    ArrayRef<Value> operands = oc.getOperands();
    hash_code operandHash;
    if (oc.isCommutative() && operands.size() == 2) {
      size_t lhsHash = hash_value(operands[0]);
      size_t rhsHash = hash_value(operands[1]);
      if (rhsHash < lhsHash) {
        std::swap(lhsHash, rhsHash);
      }
      operandHash = hash_combine(lhsHash, rhsHash);
    } else {
      operandHash = hash_combine_range(operands.begin(), operands.end());
    }

    const ReadStateKey &key = oc.getReadStateKey();
    return hash_combine(
        opHash, operandHash, key.block, key.lastCommonBarrier, key.lastStateMutation
    );
  }
  static bool isEqual(const OperationComparator &lhs, const OperationComparator &rhs) {
    return lhs == rhs;
  }
};

} // namespace llvm

namespace {

class PassImpl : public llzk::impl::RedundantOperationEliminationPassBase<PassImpl> {
  using Base = RedundantOperationEliminationPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    SymbolTableCollection symbolTables;
    // Traverse functions from the bottom of the call graph up.
    // This way, we may create empty constrain functions to which we can eliminate
    // calls.
    auto &cga = getAnalysis<CallGraphAnalysis>();
    const llzk::CallGraph *callGraph = &cga.getCallGraph();
    for (auto it = llvm::po_begin(callGraph); it != llvm::po_end(callGraph); ++it) {
      const llzk::CallGraphNode *node = *it;
      if (!node->isExternal()) {
        runOnFunc(symbolTables, node->getCalledFunction());
      }
    }
  }

  bool isPurposelessConstrainFunc(SymbolTableCollection &symbolTables, FuncDefOp fn) {
    if (!fn.isStructConstrain()) {
      return false;
    }
    // Calls to a constrain function are only removable when the callee cannot
    // contain witness-generation state mutations such as global.write or
    // ram.store. The WitnessGen verifier enforces that boundary unless the
    // callee is explicitly marked allow_witness.
    if (fn.hasAllowWitnessAttr()) {
      return false;
    }

    bool res = true;
    fn.walk([&](Operation *op) {
      if (op == fn.getOperation()) {
        return WalkResult::advance();
      }
      if (isa<EmitEqualityOp, EmitContainmentOp, AssertOp>(op)) {
        res = false;
        return WalkResult::interrupt();
      } else if (auto callOp = dyn_cast<CallOp>(op)) {
        if (!callsPurposelessConstrainFunc(symbolTables, callOp)) {
          res = false;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      } else if (isMemoryEffectFree(op)) {
        return WalkResult::advance();
      }

      // Purposeless calls skip updateReadState, so apply its conservative
      // unknown-effect policy before allowing a call to be removed.
      if (hasUnknownOrNonReadEffect(op)) {
        res = false;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return res;
  }

  bool callsPurposelessConstrainFunc(SymbolTableCollection &symbolTables, CallOp call) {
    auto callLookup = resolveCallable<FuncDefOp>(symbolTables, call);
    return succeeded(callLookup) && isPurposelessConstrainFunc(symbolTables, callLookup->get());
  }

  void runOnFunc(SymbolTableCollection &symbolTables, FuncDefOp fn) {
    TranslationMap map;
    SmallVector<Operation *> redundantOps;
    DenseSet<OperationComparator> uniqueOps;
    DenseMap<Block *, BlockReadState> readStates;
    DominanceInfo domInfo(fn);

    auto unnecessaryOpCheck = [&](Operation *op) -> bool {
      if (auto emiteq = dyn_cast<EmitEqualityOp>(op);
          emiteq && emiteq.getLhs() == emiteq.getRhs()) {
        redundantOps.push_back(op);
        return true;
      }

      if (auto callOp = dyn_cast<CallOp>(op);
          callOp && callsPurposelessConstrainFunc(symbolTables, callOp)) {
        redundantOps.push_back(op);
        return true;
      }
      return false;
    };

    fn.walk([&](Operation *op) {
      if (op == fn.getOperation()) {
        return WalkResult::advance();
      }

      // Case 1: The operation itself is unnecessary.
      if (unnecessaryOpCheck(op)) {
        return WalkResult::advance();
      }

      BlockReadState &readState = readStates[op->getBlock()];

      // Case 2: An equivalent operation A has already been performed before
      // the current operation B and A dominates B.
      bool isRedundant = false;
      if (isDuplicateEliminationCandidate(op)) {
        OperationComparator comp(op, map, getReadStateKey(op, readState));
        if (auto it = uniqueOps.find(comp);
            it != uniqueOps.end() && domInfo.dominates(it->getOp(), op)) {
          redundantOps.push_back(op);
          isRedundant = true;
          for (unsigned opNum = 0; opNum < op->getNumResults(); opNum++) {
            map[op->getResult(opNum)] = it->getOp()->getResult(opNum);
          }
        } else {
          uniqueOps.insert(comp);
        }
      }

      if (!isRedundant) {
        updateReadState(op, readState);
      }
      return WalkResult::advance();
    });

    DenseSet<Operation *> redundantOpSet;
    for (Operation *op : redundantOps) {
      redundantOpSet.insert(op);
    }

    SmallVector<Operation *> deadOpCandidates;
    DenseSet<Operation *> queuedDeadOps;
    auto enqueueDeadOpCandidate = [&](Value value) {
      Operation *definingOp = value.getDefiningOp();
      if (!definingOp || redundantOpSet.count(definingOp) ||
          !queuedDeadOps.insert(definingOp).second) {
        return;
      }
      deadOpCandidates.push_back(definingOp);
    };

    for (Operation *op : redundantOps) {
      LLVM_DEBUG(llvm::dbgs() << "Removing op: " << *op << '\n');
      for (Value result : op->getResults()) {
        if (!result.use_empty()) {
          auto it = map.find(result);
          ensure(
              it != map.end(), "failed to find a replacement value for redundant operation result"
          );
          LLVM_DEBUG(llvm::dbgs() << "Replacing " << it->first << " with " << it->second << '\n');
          result.replaceAllUsesWith(it->second);
        }
      }

      SmallVector<Value> operands(op->getOperands());
      op->erase();
      for (Value operand : operands) {
        enqueueDeadOpCandidate(operand);
      }
    }

    // Removing a redundant op may make its producers dead. Check whole
    // operations so effects and every result are considered before erasure.
    while (!deadOpCandidates.empty()) {
      Operation *op = deadOpCandidates.pop_back_val();
      queuedDeadOps.erase(op);
      if (!isDeadAfterElimination(op)) {
        continue;
      }

      SmallVector<Value> operands(op->getOperands());
      LLVM_DEBUG(llvm::dbgs() << "Removing dead producer: " << *op << '\n');
      op->erase();
      for (Value operand : operands) {
        enqueueDeadOpCandidate(operand);
      }
    }
  }
};

} // namespace
