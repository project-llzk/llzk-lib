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

#include <utility>

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
    OpPassManager &pm, polymorphic::FlatteningPassOptions flattening, bool arrayToScalar,
    bool podToScalar, std::unique_ptr<Pass> inliningPass
) {
  // default to `main-as-root` if unspecified to avoid leaving parameterized templates
  // that cause the later struct inlining pass to crash
  if (flattening.cleanupMode == polymorphic::FlatteningCleanupMode::Unspecified) {
    flattening.cleanupMode = polymorphic::FlatteningCleanupMode::MainAsRoot;
  }
  pm.addPass(polymorphic::createFlatteningPass(flattening));

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

  // Remove struct and member definitions that are no longer used after inlining.
  pm.addPass(createUnusedDeclarationEliminationPass(
      UnusedDeclarationEliminationPassOptions {.removeStructs = true}
  ));
}

void buildFullPolyLoweringPipelineImpl(
    OpPassManager &pm, polymorphic::FlatteningPassOptions flattening, bool arrayToScalar,
    bool podToScalar, std::unique_ptr<Pass> inliningPass, std::unique_ptr<Pass> polyLoweringPass
) {
  // 1. Struct flattening and inlining
  buildFullStructInliningPipelineImpl(
      pm, flattening, arrayToScalar, podToScalar, std::move(inliningPass)
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
      pm, cfg.flattening, cfg.arrayToScalar, cfg.podToScalar,
      component::createInlineStructsPass(cfg.inlining)
  );
}

void buildFullPolyLoweringPipeline(OpPassManager &pm, const FullPolyLoweringConfig &cfg) {
  buildFullPolyLoweringPipelineImpl(
      pm, cfg.structInlining.flattening, cfg.structInlining.arrayToScalar,
      cfg.structInlining.podToScalar,
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
      "Run flattening and inlining of all struct definitions into the `main` struct. This "
      "pipeline uses the `main-as-root` cleanup mode in the flattening pass by default. It "
      "is not recommended to override this cleanup mode because other cleanup modes may "
      "leave behind parameterized templates that later cause `llzk-inline-structs` to crash.",
      [](OpPassManager &pm, const FullStructInliningOptions &opts) {
    auto flattening = opts.flattening.getValue().createOptions();
    buildFullStructInliningPipelineImpl(
        pm, flattening->createPassOptions(), opts.arrayToScalar, opts.podToScalar,
        createConfiguredPass(opts.inlining)
    );
  }
  );

  PassPipelineRegistration<FullPolyLoweringOptions>(
      "llzk-full-poly-lowering",
      "Run flattening and inlining of all struct definitions into the `main` struct, then lower "
      "polynomial constraints to a given max degree, and finally remove unnecessary operations and "
      "definitions. This pipeline uses the `main-as-root` cleanup mode in the flattening pass by "
      "default. It is not recommended to override this cleanup mode because other cleanup modes "
      "may leave behind parameterized templates that later cause `llzk-inline-structs` to crash.",
      [](OpPassManager &pm, const FullPolyLoweringOptions &opts) {
    auto structInlining = opts.structInlining.getValue().createOptions();
    auto flattening = structInlining->flattening.getValue().createOptions();
    buildFullPolyLoweringPipelineImpl(
        pm, flattening->createPassOptions(), structInlining->arrayToScalar,
        structInlining->podToScalar, createConfiguredPass(structInlining->inlining),
        createConfiguredPass(opts.polyLowering)
    );
  }
  );

  PassPipelineRegistration<>(
      "llzk-verif-to-smt",
      "Normalize array/pod aggregates and lower verif contracts to SMT helpers",
      [](OpPassManager &pm) {
    pm.addPass(llzk::createAggregateScalarizationPass());
    pm.addPass(llzk::createVerifToSmtPass());
  }
  );
}

} // namespace llzk
