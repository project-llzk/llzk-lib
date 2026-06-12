//===-- LLZKRemoveUnusedDiscardableAllocationsPass.cpp ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-remove-unused-discardable-allocations` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/DiscardableAllocationOpInterfaces.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_REMOVEUNUSEDDISCARDABLEALLOCATIONSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;

namespace {

/// Returns true when the allocator is explicitly marked as safe for discardable-allocation cleanup.
static bool hasDiscardableAllocationEffect(Operation *allocator) {
  auto effectInterface = llvm::dyn_cast<MemoryEffectOpInterface>(allocator);
  if (!effectInterface) {
    return false;
  }

  llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  effectInterface.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    return llvm::isa<MemoryEffects::Allocate>(effect.getEffect()) &&
           llvm::isa<DiscardableAllocationResource>(effect.getResource());
  });
}

/// Collects direct write-only users, returning false if any user can read or retain the allocation.
static FailureOr<bool> collectRemovableUsers(
    Operation *allocator, SmallVectorImpl<Operation *> &usersToErase, const DataLayout &dataLayout
) {
  if (allocator->getNumResults() != 1) {
    allocator->emitOpError()
        << "selected for discardable-allocation cleanup but does not have one result";
    return failure();
  }

  Value allocation = allocator->getResult(0);
  if (allocation.use_empty()) {
    return true;
  }

  for (OpOperand &use : allocation.getUses()) {
    Operation *user = use.getOwner();
    if (user->mightHaveTrait<OpTrait::IsTerminator>()) {
      return false;
    }

    auto accessor = llvm::dyn_cast<DiscardableAllocationAccessorOpInterface>(user);
    if (!accessor || accessor.loadsFromDiscardableAllocation(allocation) ||
        !accessor.storesToDiscardableAllocation(allocation) ||
        !accessor.canEraseAsDeadStoreTo(allocation, dataLayout)) {
      return false;
    }
    usersToErase.push_back(user);
  }

  return true;
}

/// Collects operand definitions that may become dead after \p opsToErase are removed.
static void collectOperandDefiningOps(
    llvm::ArrayRef<Operation *> opsToErase,
    const llvm::SmallPtrSetImpl<Operation *> &opsBeingErased,
    SmallVectorImpl<Operation *> &maybeDeadDefs
) {
  llvm::SmallPtrSet<Operation *, 16> seenDefs;
  for (Operation *op : opsToErase) {
    for (Value operand : op->getOperands()) {
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp && !opsBeingErased.contains(definingOp) &&
          seenDefs.insert(definingOp).second) {
        maybeDeadDefs.push_back(definingOp);
      }
    }
  }
}

/// Erases trivially dead defining ops, including newly dead operands discovered while erasing.
static bool eraseTriviallyDeadDefs(SmallVectorImpl<Operation *> &maybeDeadDefs) {
  bool changed = false;
  bool changedThisIteration = false;
  llvm::SmallPtrSet<Operation *, 16> seen(maybeDeadDefs.begin(), maybeDeadDefs.end());
  llvm::SmallPtrSet<Operation *, 16> erased;

  do {
    changedThisIteration = false;
    for (size_t i = 0; i < maybeDeadDefs.size(); ++i) {
      Operation *op = maybeDeadDefs[i];
      if (erased.contains(op) || !isOpTriviallyDead(op)) {
        continue;
      }
      for (Value operand : op->getOperands()) {
        if (Operation *definingOp = operand.getDefiningOp()) {
          if (seen.insert(definingOp).second) {
            maybeDeadDefs.push_back(definingOp);
          }
        }
      }
      op->erase();
      erased.insert(op);
      changedThisIteration = true;
    }
    changed |= changedThisIteration;
  } while (changedThisIteration);

  return changed;
}

/// Erases selected allocators marked with `MemAlloc<DiscardableAllocationResource>` when their
/// direct users are only erasable dead stores. Any read, terminator use, non-discardable accessor
/// user, or self-store keeps the allocation and all users intact.
static FailureOr<bool> removeUnusedDiscardableAllocations(
    ModuleOp module, StringRef allocatorOpName, const DataLayout &dataLayout
) {
  bool changed = false;
  bool changedThisIteration = false;
  do {
    changedThisIteration = false;
    SmallVector<Operation *> opsToErase;
    SmallVector<Operation *> maybeDeadDefs;
    llvm::SmallPtrSet<Operation *, 16> seen;

    WalkResult walkResult = module->walk([&](Operation *allocator) {
      if (allocator->getName().getStringRef() != allocatorOpName) {
        return WalkResult::advance();
      }
      if (!hasDiscardableAllocationEffect(allocator)) {
        allocator->emitOpError()
            << "selected for discardable-allocation cleanup but is not marked with "
               "MemAlloc<DiscardableAllocationResource>";
        return WalkResult::interrupt();
      }

      SmallVector<Operation *> usersToErase;
      FailureOr<bool> canRemove = collectRemovableUsers(allocator, usersToErase, dataLayout);
      if (failed(canRemove)) {
        return WalkResult::interrupt();
      }
      if (!*canRemove) {
        return WalkResult::advance();
      }
      for (Operation *user : usersToErase) {
        if (seen.insert(user).second) {
          opsToErase.push_back(user);
        }
      }
      if (seen.insert(allocator).second) {
        opsToErase.push_back(allocator);
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      return failure();
    }

    collectOperandDefiningOps(opsToErase, seen, maybeDeadDefs);
    for (Operation *op : opsToErase) {
      op->erase();
    }

    changedThisIteration = !opsToErase.empty();
    changedThisIteration |= eraseTriviallyDeadDefs(maybeDeadDefs);
    changed |= changedThisIteration;
  } while (changedThisIteration);

  return changed;
}

class PassImpl : public llzk::impl::RemoveUnusedDiscardableAllocationsPassBase<PassImpl> {
  using Base = RemoveUnusedDiscardableAllocationsPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (allocatorOpName.empty()) {
      module.emitError() << "requires non-empty 'allocator-op' option";
      signalPassFailure();
      return;
    }

    auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
    const DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(module);

    FailureOr<bool> changed =
        removeUnusedDiscardableAllocations(module, allocatorOpName, dataLayout);
    if (failed(changed)) {
      signalPassFailure();
      return;
    }

    if (!*changed) {
      markAllAnalysesPreserved();
    }
  }
};

} // namespace
