//===-- LLZKTransformationPassPipelines.cpp ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements logic for registering several pass pipelines.
///
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKTransformationPassPipelines.h"

#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace llzk {

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

namespace {

template <typename NestedPassOptionT>
inline std::unique_ptr<Pass> createConfiguredPass(const NestedPassOptionT &options) {
  return options.getValue().createPass();
}

void buildFullStructInliningPipelineImpl(
    OpPassManager &pm, std::unique_ptr<Pass> flatteningPass, bool arrayToScalar, bool podToScalar,
    std::unique_ptr<Pass> inliningPass
) {
  pm.addPass(std::move(flatteningPass));

  // Run array-to-scalar first because it can split arrays within a pod
  // but pod-to-scalar cannot split pods within an array.
  if (arrayToScalar) {
    pm.addPass(array::createArrayToScalarPass());
  }
  if (podToScalar) {
    pm.addPass(pod::createPodToScalarPass());
  }
  // Canonicalize to remove known-condition `scf.if` regions so struct inlining
  // can link "@compute" calls to struct members.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::move(inliningPass));
}

void buildFullPolyLoweringPipelineImpl(
    OpPassManager &pm, std::unique_ptr<Pass> flatteningPass, bool arrayToScalar, bool podToScalar,
    std::unique_ptr<Pass> inliningPass, std::unique_ptr<Pass> polyLoweringPass
) {
  // 1. Struct flattening and inlining
  buildFullStructInliningPipelineImpl(
      pm, std::move(flatteningPass), arrayToScalar, podToScalar, std::move(inliningPass)
  );
  // 2. Degree lowering
  pm.addPass(std::move(polyLoweringPass));
  // 3. Cleanup
  buildRemoveUnnecessaryOpsAndDefsPipeline(pm);
}

} // namespace

void buildRemoveUnnecessaryOpsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createRedundantReadAndWriteEliminationPass());
  pm.addPass(createRedundantOperationEliminationPass());
}

void buildRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &pm) {
  buildRemoveUnnecessaryOpsPipeline(pm);
  pm.addPass(createUnusedDeclarationEliminationPass());
}

void buildProductProgramPipeline(OpPassManager &pm) {
  pm.addPass(createComputeConstrainToProductPass());
  pm.addPass(createFuseProductLoopsPass());
}

void buildFullStructInliningPipeline(OpPassManager &pm, const FullStructInliningConfig &cfg) {
  buildFullStructInliningPipelineImpl(
      pm, polymorphic::createFlatteningPass(cfg.flattening), cfg.arrayToScalar, cfg.podToScalar,
      component::createInlineStructsPass(cfg.inlining)
  );
}

void buildFullPolyLoweringPipeline(OpPassManager &pm, const FullPolyLoweringConfig &cfg) {
  buildFullPolyLoweringPipelineImpl(
      pm, polymorphic::createFlatteningPass(cfg.structInlining.flattening),
      cfg.structInlining.arrayToScalar, cfg.structInlining.podToScalar,
      component::createInlineStructsPass(cfg.structInlining.inlining),
      createPolyLoweringPass(cfg.polyLowering)
  );
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops",
      "Remove unnecessary operations, such as redundant reads or repeated constraints",
      buildRemoveUnnecessaryOpsPipeline
  );

  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops-and-defs",
      "Remove unnecessary operations, member definitions, and struct definitions",
      buildRemoveUnnecessaryOpsAndDefsPipeline
  );

  PassPipelineRegistration<>(
      "llzk-product-program",
      "Convert @compute/@constrain functions to @product function and perform alignment",
      buildProductProgramPipeline
  );

  PassPipelineRegistration<FullStructInliningOptions>(
      "llzk-full-struct-inlining",
      "Run flattening and inlining of all struct definitions into the `main` struct.",
      [](OpPassManager &pm, const FullStructInliningOptions &opts) {
    buildFullStructInliningPipelineImpl(
        pm, createConfiguredPass(opts.flattening), opts.arrayToScalar, opts.podToScalar,
        createConfiguredPass(opts.inlining)
    );
  }
  );

  PassPipelineRegistration<FullPolyLoweringOptions>(
      "llzk-full-poly-lowering",
      "Lower polynomial constraints to a given max degree, then remove "
      "unnecessary operations and definitions.",
      [](OpPassManager &pm, const FullPolyLoweringOptions &opts) {
    auto structInlining = opts.structInlining.getValue().createOptions();
    buildFullPolyLoweringPipelineImpl(
        pm, createConfiguredPass(structInlining->flattening), structInlining->arrayToScalar,
        structInlining->podToScalar, createConfiguredPass(structInlining->inlining),
        createConfiguredPass(opts.polyLowering)
    );
  }
  );
}

} // namespace llzk
