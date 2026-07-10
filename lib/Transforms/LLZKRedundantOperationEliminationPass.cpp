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
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallVector.h>

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

static auto EMPTY_OP_KEY = reinterpret_cast<Operation *>(1);
static auto TOMBSTONE_OP_KEY = reinterpret_cast<Operation *>(2);

// Maps original -> replacement value
using TranslationMap = DenseMap<Value, Value>;

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

  OperationComparator(Operation *o, const TranslationMap &m) : op(o) {
    for (auto operand : op->getOperands()) {
      if (auto it = m.find(operand); it != m.end()) {
        operands.push_back(it->second);
      } else {
        operands.push_back(operand);
      }
    }
  }

  Operation *getOp() const { return op; }

  const SmallVector<Value> &getOperands() const { return operands; }

  bool isCommutative() const { return op->hasTrait<OpTrait::IsCommutative>(); }

  friend bool operator==(const OperationComparator &lhs, const OperationComparator &rhs) {
    if (lhs.op == EMPTY_OP_KEY || rhs.op == EMPTY_OP_KEY || lhs.op == TOMBSTONE_OP_KEY ||
        rhs.op == TOMBSTONE_OP_KEY) {
      return lhs.op == rhs.op;
    }

    if (lhs.op->getName() != rhs.op->getName()) {
      return false;
    }

    // uninterested in operating over control-flow ops
    auto dialectName = lhs.op->getDialect()->getNamespace();
    if (dialectName == scf::SCFDialect::getDialectNamespace()) {
      return false;
    }

    // This may be overly restrictive in some cases, but without knowing what
    // potential future attributes we may have, it's safer to assume that
    // unequal attributes => unequal operations.
    // This covers constant operations too, as the constant is an attribute,
    // not an operand.
    if (lhs.op->getAttrs() != rhs.op->getAttrs()) {
      return false;
    }
    // For commutative operations, just check if the operands contain the same set in any order
    if (lhs.isCommutative()) {
      ensure(
          lhs.operands.size() == 2 && rhs.operands.size() == 2,
          "No known commutative ops have more than two arguments"
      );
      return (lhs.operands[0] == rhs.operands[0] && lhs.operands[1] == rhs.operands[1]) ||
             (lhs.operands[0] == rhs.operands[1] && lhs.operands[1] == rhs.operands[0]);
    }

    // The default case requires an exact match per argument
    return lhs.operands == rhs.operands;
  }

private:
  Operation *op;
  SmallVector<Value> operands;
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
    // Just hash on name to force more thorough equality checks by operation type.
    return hash_value(oc.getOp()->getName());
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

    bool res = true;
    fn.walk([&](Operation *op) {
      if (isa<EmitEqualityOp, EmitContainmentOp, AssertOp>(op)) {
        res = false;
        return WalkResult::interrupt();
      } else if (auto callOp = dyn_cast<CallOp>(op);
                 callOp && !callsPurposelessConstrainFunc(symbolTables, callOp)) {
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
      // Case 1: The operation itself is unnecessary.
      if (unnecessaryOpCheck(op)) {
        return WalkResult::advance();
      }

      // Case 2: An equivalent operation A has already been performed before
      // the current operation B and A dominates B.
      if (!isa<NonDetOp, AuxOp>(op)) {
        OperationComparator comp(op, map);
        if (auto it = uniqueOps.find(comp);
            it != uniqueOps.end() && domInfo.dominates(it->getOp(), op)) {
          redundantOps.push_back(op);
          for (unsigned opNum = 0; opNum < op->getNumResults(); opNum++) {
            map[op->getResult(opNum)] = it->getOp()->getResult(opNum);
          }
        } else {
          uniqueOps.insert(comp);
        }
      }

      return WalkResult::advance();
    });

    SmallVector<Operation *> unusedOps;
    DenseSet<Operation *> queuedUnusedOps;
    auto enqueueUnusedDef = [&](Value value) {
      if (Operation *definingOp = value.getDefiningOp();
          definingOp && queuedUnusedOps.insert(definingOp).second) {
        unusedOps.push_back(definingOp);
      }
    };

    for (auto *op : redundantOps) {
      LLVM_DEBUG(llvm::dbgs() << "Removing op: " << *op << '\n');
      for (auto result : op->getResults()) {
        if (!result.getUsers().empty()) {
          auto it = map.find(result);
          ensure(
              it != map.end(), "failed to find a replacement value for redundant operation result"
          );
          LLVM_DEBUG(llvm::dbgs() << "Replacing " << it->first << " with " << it->second << '\n');
          result.replaceAllUsesWith(it->second);
        }
      }
      for (Value operand : op->getOperands()) {
        enqueueUnusedDef(operand);
      }
      op->erase();
    }

    while (!unusedOps.empty()) {
      Operation *op = unusedOps.pop_back_val();
      // Member reads have no observable effect, but do not implement MLIR's
      // MemoryEffectOpInterface and are therefore not trivially dead.
      if (!isOpTriviallyDead(op) && !(isa<MemberReadOp>(op) && op->use_empty())) {
        continue;
      }
      for (Value operand : op->getOperands()) {
        enqueueUnusedDef(operand);
      }
      LLVM_DEBUG(llvm::dbgs() << "Removing unused operation: " << *op << '\n');
      op->erase();
    }
  }
};

} // namespace
