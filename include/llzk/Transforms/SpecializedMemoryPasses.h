//===-- SpecializedMemoryPasses.h - Targeted memory passes -------*- C++ -*-===//
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

/// A pass that erases allocator ops of type \p AllocOpTy when their remaining uses are removable
/// stores. Use this only for allocator ops whose allocation has no externally observable effect
/// unless its stored values are read or escaped.
template <typename AllocOpTy>
struct SpecializedRemoveUnusedAllocations
    : mlir::PassWrapper<SpecializedRemoveUnusedAllocations<AllocOpTy>, mlir::OperationPass<>> {

  mlir::StringRef getArgument() const override {
    return "llzk-specialized-remove-unused-allocations";
  }

  mlir::StringRef getDescription() const override {
    return "Removes unread allocation ops of a specific allocator op type";
  }

  void runOnOperation() override {
    bool changed = false;
    bool changedThisIteration = false;
    do {
      changedThisIteration = false;
      mlir::SmallVector<mlir::Operation *> opsToErase;
      llvm::SmallPtrSet<mlir::Operation *, 16> seen;

      this->getOperation()->walk([&](AllocOpTy allocator) {
        mlir::SmallVector<mlir::Operation *> usersToErase;
        if (!collectRemovableUsers(allocator, usersToErase)) {
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

      for (mlir::Operation *op : opsToErase) {
        op->erase();
      }

      changedThisIteration = !opsToErase.empty();
      changed |= changedThisIteration;
    } while (changedThisIteration);

    if (!changed) {
      this->markAllAnalysesPreserved();
    }
  }

private:
  static bool collectRemovableUsers(
      AllocOpTy allocator, mlir::SmallVectorImpl<mlir::Operation *> &usersToErase
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
      if (!memOp || !isRemovableStore(memOp, slots)) {
        return false;
      }
      usersToErase.push_back(user);
    }

    return true;
  }

  static bool
  isRemovableStore(mlir::PromotableMemOpInterface memOp, llvm::ArrayRef<mlir::MemorySlot> slots) {
    return llvm::any_of(slots, [&](mlir::MemorySlot slot) {
      return memOp.storesTo(slot) && !memOp.loadsFrom(slot);
    });
  }
};

/// Pass factory for `SpecializedRemoveUnusedAllocations`.
template <typename AllocOpTy>
std::unique_ptr<SpecializedRemoveUnusedAllocations<AllocOpTy>>
createSpecializedRemoveUnusedAllocationsPass() {
  return std::make_unique<SpecializedRemoveUnusedAllocations<AllocOpTy>>();
}

} // namespace llzk
