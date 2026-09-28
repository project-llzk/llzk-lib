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

void buildFullR1CSLoweringPipeline(OpPassManager &pm, R1CSLoweringMode mode) {
  if (mode == R1CSLoweringMode::Direct) {
    pm.addPass(llzk::createPolyLoweringPass(llzk::PolyLoweringPassOptions {.maxDegree = 2}));
  } else {
    llzk::FullPolyLoweringConfig config;
    config.polyLowering = llzk::PolyLoweringPassOptions {.maxDegree = 2};
    llzk::buildFullPolyLoweringPipeline(pm, config);
  }
  pm.addPass(createR1CSPreparePass());
  if (mode == R1CSLoweringMode::Direct) {
    pm.addPass(createR1CSDirectLoweringPass());
  } else {
    pm.addPass(createR1CSLoweringPass());
  }
  pm.addPass(mlir::createCSEPass());
}

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-full-r1cs-lowering", "Lower legacy polynomial constraints to R1CS",
      [](OpPassManager &pm) { buildFullR1CSLoweringPipeline(pm, R1CSLoweringMode::Legacy); }
  );
  PassPipelineRegistration<>(
      "llzk-full-direct-r1cs-lowering", "Lower evaluated storage constraints directly to R1CS",
      [](OpPassManager &pm) { buildFullR1CSLoweringPipeline(pm, R1CSLoweringMode::Direct); }
  );
}

} // namespace r1cs
