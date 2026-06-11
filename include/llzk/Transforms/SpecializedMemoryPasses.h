//===-- SpecializedMemoryPasses.h - Targeted memory passes ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides pass templates that restrict allocation-op walks to a single concrete op type rather
/// than collecting every op that implements the corresponding allocation interface.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/Mem2Reg.h>
#include <mlir/Transforms/SROA.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>

namespace llzk {

/// A variant of the MLIR `sroa` pass that only destructures memory slots belonging to allocators
/// of type \p AllocOpTy (which must implement `mlir::DestructurableAllocationOpInterface`). All
/// other allocator ops are ignored. The rest of the pass body is identical to the upstream pass.
template <typename AllocOpTy>
struct SpecializedSROA : mlir::PassWrapper<SpecializedSROA<AllocOpTy>, mlir::OperationPass<>> {

  mlir::StringRef getArgument() const override { return "llzk-specialized-sroa"; }

  mlir::StringRef getDescription() const override {
    return "Scalar replacement of aggregates for a specific allocator op type";
  }

  void runOnOperation() override {
    mlir::Operation *scopeOp = this->getOperation();

    auto &dataLayoutAnalysis = this->template getAnalysis<mlir::DataLayoutAnalysis>();
    const mlir::DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(scopeOp);

    bool changed = false;

    for (mlir::Region &region : scopeOp->getRegions()) {
      if (region.getBlocks().empty()) {
        continue;
      }

      mlir::OpBuilder builder(&region.front(), region.front().begin());

      mlir::SmallVector<mlir::DestructurableAllocationOpInterface> allocators;
      region.walk([&](AllocOpTy allocator) { allocators.emplace_back(allocator); });

      if (mlir::succeeded(mlir::tryToDestructureMemorySlots(allocators, builder, dataLayout))) {
        changed = true;
      }
    }

    if (!changed) {
      this->markAllAnalysesPreserved();
    }
  }
};

// Pass factory for `SpecializedSROA`.
template <typename AllocOpTy>
std::unique_ptr<SpecializedSROA<AllocOpTy>> createSpecializedSROAPass() {
  return std::make_unique<SpecializedSROA<AllocOpTy>>();
}

/// A variant of the MLIR `mem2reg` pass that only promotes memory slots belonging to allocators
/// of type \p AllocOpTy (which must implement `mlir::PromotableAllocationOpInterface`). All
/// other allocator ops are ignored. The rest of the pass body is identical to the upstream pass.
template <typename AllocOpTy>
struct SpecializedMem2Reg
    : mlir::PassWrapper<SpecializedMem2Reg<AllocOpTy>, mlir::OperationPass<>> {

  mlir::StringRef getArgument() const override { return "llzk-specialized-mem2reg"; }

  mlir::StringRef getDescription() const override {
    return "Promotes memory slots of a specific allocator op type into values";
  }

  void runOnOperation() override {
    mlir::Operation *scopeOp = this->getOperation();

    auto &dataLayoutAnalysis = this->template getAnalysis<mlir::DataLayoutAnalysis>();
    const mlir::DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(scopeOp);
    auto &dominance = this->template getAnalysis<mlir::DominanceInfo>();

    bool changed = false;

    for (mlir::Region &region : scopeOp->getRegions()) {
      if (region.getBlocks().empty()) {
        continue;
      }

      mlir::OpBuilder builder(&region.front(), region.front().begin());

      mlir::SmallVector<mlir::PromotableAllocationOpInterface> allocators;
      region.walk([&](AllocOpTy allocator) { allocators.emplace_back(allocator); });

      if (mlir::succeeded(
              mlir::tryToPromoteMemorySlots(allocators, builder, dataLayout, dominance)
          )) {
        changed = true;
      }
    }

    if (!changed) {
      this->markAllAnalysesPreserved();
    }
  }
};

// Pass factory for `SpecializedMem2Reg`.
template <typename AllocOpTy>
std::unique_ptr<SpecializedMem2Reg<AllocOpTy>> createSpecializedMem2RegPass() {
  return std::make_unique<SpecializedMem2Reg<AllocOpTy>>();
}

/// Erases allocators of type \p AllocOpTy that are only stored to and never loaded from.
///
/// The allocator must have a `MemAlloc<ResourceTy>` effect. Any read, terminator use,
/// non-promotable user, or memory op that cannot remove its use keeps the allocator and all users
/// intact.
template <typename AllocOpTy, typename ResourceTy>
struct SpecializedRemoveUnusedAllocations
    : mlir::PassWrapper<
          SpecializedRemoveUnusedAllocations<AllocOpTy, ResourceTy>, mlir::OperationPass<>> {

  mlir::StringRef getArgument() const override {
    return "llzk-specialized-remove-unused-allocations";
  }

  mlir::StringRef getDescription() const override {
    return "Removes unread allocation ops of a specific allocator op type";
  }

  void runOnOperation() override {
    mlir::Operation *scopeOp = this->getOperation();
    auto &dataLayoutAnalysis = this->template getAnalysis<mlir::DataLayoutAnalysis>();
    const mlir::DataLayout &dataLayout = dataLayoutAnalysis.getAtOrAbove(scopeOp);

    bool changed = false;
    bool changedThisIteration = false;
    do {
      changedThisIteration = false;
      mlir::SmallVector<mlir::Operation *> opsToErase;
      mlir::SmallVector<mlir::Operation *> maybeDeadDefs;
      llvm::SmallPtrSet<mlir::Operation *, 16> seen;

      scopeOp->walk([&](AllocOpTy allocator) {
        mlir::SmallVector<mlir::Operation *> usersToErase;
        if (!hasRequiredAllocationEffect(allocator) ||
            !collectRemovableUsers(allocator, usersToErase, dataLayout)) {
          return;
        }
        for (mlir::Operation *user : usersToErase) {
          if (seen.insert(user).second) {
            opsToErase.push_back(user);
          }
        }
        if (seen.insert(allocator).second) {
          opsToErase.push_back(allocator);
        }
      });

      collectOperandDefiningOps(opsToErase, seen, maybeDeadDefs);
      for (mlir::Operation *op : opsToErase) {
        op->erase();
      }

      changedThisIteration = !opsToErase.empty();
      changedThisIteration |= eraseTriviallyDeadDefs(maybeDeadDefs);
      changed |= changedThisIteration;
    } while (changedThisIteration);

    if (!changed) {
      this->markAllAnalysesPreserved();
    }
  }

private:
  /// Returns true when the allocator is explicitly marked as safe for this cleanup pass.
  static bool hasRequiredAllocationEffect(AllocOpTy allocator) {
    auto effectInterface = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(allocator.getOperation());
    if (!effectInterface) {
      return false;
    }

    llvm::SmallVector<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>, 4> effects;
    effectInterface.getEffects(effects);
    return llvm::any_of(effects, [](const auto &effect) {
      return llvm::isa<mlir::MemoryEffects::Allocate>(effect.getEffect()) &&
             llvm::isa<ResourceTy>(effect.getResource());
    });
  }

  /// Collects direct write-only users, returning false if any user can read or retain the
  /// allocation.
  static bool collectRemovableUsers(
      AllocOpTy allocator, mlir::SmallVectorImpl<mlir::Operation *> &usersToErase,
      const mlir::DataLayout &dataLayout
  ) {
    if (allocator->use_empty()) {
      return true;
    }

    mlir::SmallVector<mlir::MemorySlot> slots = allocator.getPromotableSlots();
    if (slots.empty()) {
      return false;
    }

    for (mlir::OpOperand &use : allocator->getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user->mightHaveTrait<mlir::OpTrait::IsTerminator>()) {
        return false;
      }

      auto memOp = llvm::dyn_cast<mlir::PromotableMemOpInterface>(user);
      if (!memOp || !canRemoveUse(memOp, use, slots, dataLayout)) {
        return false;
      }
      usersToErase.push_back(user);
    }

    return true;
  }

  /// Returns true when \p use is the only blocking use of a removable store for a candidate slot.
  static bool canRemoveUse(
      mlir::PromotableMemOpInterface memOp, mlir::OpOperand &use,
      llvm::ArrayRef<mlir::MemorySlot> slots, const mlir::DataLayout &dataLayout
  ) {
    llvm::SmallPtrSet<mlir::OpOperand *, 1> blockingUses;
    blockingUses.insert(&use);

    return llvm::any_of(slots, [&](mlir::MemorySlot slot) {
      mlir::SmallVector<mlir::OpOperand *> newBlockingUses;
      return memOp.storesTo(slot) && !memOp.loadsFrom(slot) &&
             memOp.canUsesBeRemoved(slot, blockingUses, newBlockingUses, dataLayout) &&
             newBlockingUses.empty();
    });
  }

  /// Collects operand definitions that may become dead after \p opsToErase are removed.
  static void collectOperandDefiningOps(
      llvm::ArrayRef<mlir::Operation *> opsToErase,
      const llvm::SmallPtrSetImpl<mlir::Operation *> &opsBeingErased,
      mlir::SmallVectorImpl<mlir::Operation *> &maybeDeadDefs
  ) {
    llvm::SmallPtrSet<mlir::Operation *, 16> seenDefs;
    for (mlir::Operation *op : opsToErase) {
      for (mlir::Value operand : op->getOperands()) {
        mlir::Operation *definingOp = operand.getDefiningOp();
        if (definingOp && !opsBeingErased.contains(definingOp) &&
            seenDefs.insert(definingOp).second) {
          maybeDeadDefs.push_back(definingOp);
        }
      }
    }
  }

  /// Erases trivially dead defining ops, including newly dead operands discovered while erasing.
  static bool eraseTriviallyDeadDefs(mlir::SmallVectorImpl<mlir::Operation *> &maybeDeadDefs) {
    bool changed = false;
    bool changedThisIteration = false;
    llvm::SmallPtrSet<mlir::Operation *, 16> seen(maybeDeadDefs.begin(), maybeDeadDefs.end());
    llvm::SmallPtrSet<mlir::Operation *, 16> erased;

    do {
      changedThisIteration = false;
      for (size_t i = 0; i < maybeDeadDefs.size(); ++i) {
        mlir::Operation *op = maybeDeadDefs[i];
        if (erased.contains(op) || !mlir::isOpTriviallyDead(op)) {
          continue;
        }
        for (mlir::Value operand : op->getOperands()) {
          if (mlir::Operation *definingOp = operand.getDefiningOp()) {
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
};

/// Pass factory for `SpecializedRemoveUnusedAllocations`.
template <typename AllocOpTy, typename ResourceTy>
std::unique_ptr<SpecializedRemoveUnusedAllocations<AllocOpTy, ResourceTy>>
createSpecializedRemoveUnusedAllocationsPass() {
  return std::make_unique<SpecializedRemoveUnusedAllocations<AllocOpTy, ResourceTy>>();
}

} // namespace llzk
