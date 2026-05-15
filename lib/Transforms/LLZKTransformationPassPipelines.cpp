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

#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace llzk {

struct FullPolyLoweringOptions : public PassPipelineOptions<FullPolyLoweringOptions> {
  Option<unsigned> maxDegree {
      *this, "max-degree", llvm::cl::desc("Maximum polynomial degree (must be ≥ 2)"),
      llvm::cl::init(2)
  };
};

void addRemoveUnnecessaryOpsAndDefsPipeline(OpPassManager &pm) {
  pm.addPass(llzk::createRedundantReadAndWriteEliminationPass());
  pm.addPass(llzk::createRedundantOperationEliminationPass());
  pm.addPass(llzk::createUnusedDeclarationEliminationPass());
}

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops",
      "Remove unnecessary operations, such as redundant reads or repeated constraints",
      [](OpPassManager &pm) {
    pm.addPass(createRedundantReadAndWriteEliminationPass());
    pm.addPass(createRedundantOperationEliminationPass());
  }
  );

  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops-and-defs",
      "Remove unnecessary operations, member definitions, and struct definitions",
      [](OpPassManager &pm) { addRemoveUnnecessaryOpsAndDefsPipeline(pm); }
  );

  PassPipelineRegistration<FullPolyLoweringOptions>(
      "llzk-full-poly-lowering",
      "Lower all polynomial constraints to a given max degree, then remove unnecessary operations "
      "and definitions.",
      [](OpPassManager &pm, const FullPolyLoweringOptions &opts) {
    // 1. Degree lowering
    pm.addPass(llzk::createPolyLoweringPass(opts.maxDegree));

    // 2. Cleanup
    addRemoveUnnecessaryOpsAndDefsPipeline(pm);
  }
  );

  PassPipelineRegistration<>(
      "llzk-product-program",
      "Convert @compute/@constrain functions to @product function and perform alignment",
      [](OpPassManager &pm) {
    pm.addPass(llzk::createComputeConstrainToProductPass());
    pm.addPass(llzk::createFuseProductLoopsPass());
  }
  );
}

} // namespace llzk
