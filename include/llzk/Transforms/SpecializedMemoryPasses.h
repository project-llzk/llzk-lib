//===-- SpecializedMemoryPasses.h - Targeted SROA / mem2reg -----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Provides `SpecializedSROA<AllocOpTy>` and `SpecializedMem2Reg<AllocOpTy>`:
/// pass templates that replicate the bodies of the MLIR `sroa` and `mem2reg`
/// passes but restrict the allocation-op walk to a single concrete op type rather
/// than collecting every op that implements the corresponding allocation interface.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Mem2Reg.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/Transforms/SROA.h>

#include <llvm/ADT/SmallVector.h>

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

namespace detail {

/// A workaround wrapper around MLIR's `remove-dead-values` pass that normalizes empty
/// `scf.if` else regions before running the upstream implementation and cleans up the trivial
/// regions afterwards.
class RemoveDeadValuesWorkaroundPass
    : public mlir::PassWrapper<RemoveDeadValuesWorkaroundPass, mlir::OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RemoveDeadValuesWorkaroundPass)

  llvm::StringRef getArgument() const override { return "remove-dead-values"; }
  llvm::StringRef getDescription() const override { return "Remove dead values"; }

  void runOnOperation() final {
    // Pre-pass: add a trivial block to empty `else` regions so upstream pass code can handle them.
    getOperation()->walk([](mlir::scf::IfOp ifOp) {
      if (ifOp.getElseRegion().empty()) {
        mlir::Block &elseBlock = ifOp.getElseRegion().emplaceBlock();
        mlir::OpBuilder builder(ifOp.getContext());
        builder.setInsertionPointToEnd(&elseBlock);
        builder.create<mlir::scf::YieldOp>(ifOp.getLoc());
      }
    });

    mlir::OpPassManager pm(getOperation()->getName().getStringRef());
    pm.addPass(mlir::createRemoveDeadValuesPass());
    if (mlir::failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    // Post-pass: remove trivial `else` blocks that are left behind.
    getOperation()->walk([](mlir::scf::IfOp ifOp) {
      if (ifOp.getResults().empty()) {
        mlir::Region &elseRegion = ifOp.getElseRegion();
        if (!llvm::hasSingleElement(elseRegion)) {
          return;
        }
        mlir::Block &elseBlock = elseRegion.front();
        if (!llvm::hasSingleElement(elseBlock)) {
          return;
        }
        if (!llvm::isa<mlir::scf::YieldOp>(elseBlock.front())) {
          return;
        }
        elseRegion.dropAllReferences();
        elseBlock.clear();
        elseRegion.getBlocks().clear();
      }
    });
  }
};

} // namespace detail

inline std::unique_ptr<mlir::Pass> createRemoveDeadValuesWorkaroundPass() {
  return std::make_unique<detail::RemoveDeadValuesWorkaroundPass>();
}

} // namespace llzk
