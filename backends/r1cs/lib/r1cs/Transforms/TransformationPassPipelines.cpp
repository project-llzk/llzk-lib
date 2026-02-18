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

#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "r1cs/Transforms/TransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

using namespace mlir;

namespace r1cs {

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-full-r1cs-lowering", "Lower all polynomial constraints to r1cs", [](OpPassManager &pm) {
    // 1. Degree lowering
    pm.addPass(llzk::createPolyLoweringPass(2));

    // 2. Cleanup
    llzk::addRemoveUnnecessaryOpsAndDefsPipeline(pm);

    // 3. Convert to R1CS
    pm.addPass(r1cs::createR1CSLoweringPass());

    // 4. Run CSE to eliminate to_linear ops
    pm.addPass(mlir::createCSEPass());
  }
  );
}

} // namespace r1cs
