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

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace llzk {

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void buildRemoveUnnecessaryOpsPipeline(mlir::OpPassManager &pm) {
  pm.addPass(createRedundantReadAndWriteEliminationPass());
  pm.addPass(createRedundantOperationEliminationPass());
}

void buildRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &pm) {
  buildRemoveUnnecessaryOpsPipeline(pm);
  pm.addPass(createUnusedDeclarationEliminationPass());
}

void buildFullPolyLoweringPipeline(OpPassManager &pm, const FullPolyLoweringOptions &opts) {
  // 1. Degree lowering
  pm.addPass(createPolyLoweringPass(PolyLoweringPassOptions {.maxDegree = opts.maxDegree}));
  // 2. Cleanup
  buildRemoveUnnecessaryOpsAndDefsPipeline(pm);
}

void buildProductProgramPipeline(OpPassManager &pm) {
  pm.addPass(createComputeConstrainToProductPass());
  pm.addPass(createFuseProductLoopsPass());
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

  PassPipelineRegistration<FullPolyLoweringOptions>(
      "llzk-full-poly-lowering",
      "Lower already-flattened polynomial constraints to a given max degree, then remove "
      "unnecessary operations and definitions.",
      buildFullPolyLoweringPipeline
  );

  PassPipelineRegistration<>(
      "llzk-product-program",
      "Convert @compute/@constrain functions to @product function and perform alignment",
      buildProductProgramPipeline
  );
}

} // namespace llzk
