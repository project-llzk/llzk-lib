//===-- TransformationPassPipelines.cpp -------------------------*- C++ -*-===//
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

#include "r1cs/Transforms/TransformationPassPipelines.h"

#include "r1cs/Transforms/TransformationPasses.h"

#include "llzk/Transforms/LLZKTransformationPassPipelines.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace r1cs {

void buildFullR1CSLoweringPipeline(OpPassManager &pm) {
  // 1. Polynomial degree lowering and cleanup
  llzk::FullPolyLoweringConfig config;
  config.polyLowering = llzk::PolyLoweringPassOptions {.maxDegree = 2};
  llzk::buildFullPolyLoweringPipeline(pm, config);

  // 2. Convert to R1CS
  pm.addPass(r1cs::createR1CSLoweringPass());

  // 3. Run CSE to eliminate to_linear ops
  pm.addPass(mlir::createCSEPass());

  // Other passes that may be helpful to add in the future:
  // - llzk::createRemoveDeadValuesWorkaroundPass()
  // - mlir::createCanonicalizerPass()
  //    (was run via poly-lowering -> struct-inlining but again may be useful)
}

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-full-r1cs-lowering", "Lower polynomial constraints to r1cs",
      buildFullR1CSLoweringPipeline
  );
}

} // namespace r1cs
