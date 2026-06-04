//===-- PodToScalarPass.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-pod-to-scalar` pass.
///
/// The steps of this transformation are as follows:
///
/// 0. Scan to find `llzk.nondet` ops that allocate uninitialized pods and replace them with
///    an equivalent `pod.new`
///
/// 1. Run MLIR "sroa" pass to split each pod with `N` records into `N` pods with 1 record each
///    (to prepare for "mem2reg" pass because it's API cannot deal with splitting up memory).
///
/// 2. Run MLIR "mem2reg" pass to convert all single-record pod allocations and accesses into SSA
///    values. This pass also runs several standard optimizations so the final result is condensed.
///
/// Note: This transformation imposes a "last write wins" semantics on pod records. If
/// different/configurable semantics are added in the future, some additional transformation would
/// be necessary before/during this pass so that multiple writes to the same record can be handled
/// properly while they still exist.
///
/// Note: This transformation will introduce a `nondet` op when there exists a read from a pod
/// record that was not earlier written to.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk::pod {
#define GEN_PASS_DEF_PODTOSCALARPASS
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h.inc"
} // namespace llzk::pod

using namespace mlir;
using namespace llzk;
using namespace llzk::pod;

#define DEBUG_TYPE "llzk-pod-to-scalar"

namespace {

static void baseTargetSetup(ConversionTarget &target) {
  target.addLegalDialect<
      LLZKDialect, array::ArrayDialect, boolean::BoolDialect, cast::CastDialect,
      constrain::ConstrainDialect, component::StructDialect, felt::FeltDialect,
      function::FunctionDialect, global::GlobalDialect, include::IncludeDialect, pod::PODDialect,
      polymorphic::PolymorphicDialect, ram::RAMDialect, string::StringDialect, arith::ArithDialect,
      scf::SCFDialect>();
  target.addLegalOp<ModuleOp>();
}

class NondetToNewPod : public OpConversionPattern<NonDetOp> {
  using OpConversionPattern<NonDetOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      NonDetOp nondetOp, OpAdaptor, ConversionPatternRewriter &rewriter
  ) const override {
    if (auto at = dyn_cast<PodType>(nondetOp.getType())) {
      rewriter.replaceOpWithNewOp<NewPodOp>(nondetOp, at);
      return success();
    }
    return failure();
  }
};

/// Prepare the module by replacing `llzk.nondet` pod allocation ops with `pod.new`.
static LogicalResult step0(ModuleOp modOp) {
  MLIRContext *ctx = modOp.getContext();
  RewritePatternSet patterns {ctx};
  patterns.add<NondetToNewPod>(ctx);
  ConversionTarget target {*ctx};

  baseTargetSetup(target);
  target.addDynamicallyLegalOp<NonDetOp>([](NonDetOp op) { return !isa<PodType>(op.getType()); });

  return applyFullConversion(modOp, target, std::move(patterns));
}

class PodToScalarPass : public llzk::pod::impl::PodToScalarPassBase<PodToScalarPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(step0(module))) {
      return signalPassFailure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After step 0:\n";
      module.dump();
    });

    OpPassManager nestedPM(ModuleOp::getOperationName());
    // Use SROA (Destructurable* interfaces) to split each pod with `N` records into `N` pods with 1
    // record each. This is necessary because the mem2reg pass cannot deal with splitting up memory,
    // i.e., it can only convert scalar memory access into SSA values.
    nestedPM.addPass(createSpecializedSROAPass<NewPodOp>());
    // The mem2reg pass converts the size 1 pod allocations and accesses into SSA values.
    nestedPM.addPass(createSpecializedMem2RegPass<NewPodOp>());
    // Cleanup SSA values made dead by the transformations
    nestedPM.addPass(createRemoveDeadValuesPass());
    if (failed(runPipeline(nestedPM, module))) {
      signalPassFailure();
      return;
    }
    LLVM_DEBUG({
      llvm::dbgs() << "After SROA+Mem2Reg pipeline:\n";
      module.dump();
    });
  }
};

} // namespace

std::unique_ptr<Pass> llzk::pod::createPodToScalarPass() {
  return std::make_unique<PodToScalarPass>();
};
