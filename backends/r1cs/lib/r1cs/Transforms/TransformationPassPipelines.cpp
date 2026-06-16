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
  // // TODO: may need to override some things because the defaults here are
  // // correct but the defaults in poly lowering may not be the same.
  // config.structInlining = llzk::FullStructInliningConfig {
  //     .flattening = polymorphic::FlatteningPassOptions {},
  //     .arrayToScalar = true,
  //     .podToScalar = true,
  //     .inlining = component::InlineStructsPassOptions {}
  // };
  llzk::buildFullPolyLoweringPipeline(pm, config);

  // TODO: need to remove scf control flow ops... probably canon.

  // 2. Convert to R1CS
  pm.addPass(r1cs::createR1CSLoweringPass());

  // 3. Run CSE to eliminate to_linear ops
  pm.addPass(mlir::createCSEPass());
}

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-full-r1cs-lowering", "Lower polynomial constraints to r1cs",
      buildFullR1CSLoweringPipeline
  );
}

} // namespace r1cs
