//===-- ConversionPassPipelines.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Defines the preprocessing and conversion pipeline for the PCL backend.
//===----------------------------------------------------------------------===//

#include "pcl/Conversion/ConversionPassPipelines.h"

#include "pcl/Conversion/ConversionPasses.h"

#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

namespace pcl {

void buildFullPCLLoweringPipeline(mlir::OpPassManager &pm) {
  pm.addPass(llzk::createWhileToForPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(
      llzk::polymorphic::createFlatteningPass(
          llzk::polymorphic::FlatteningPassOptions {
              .cleanupMode = llzk::polymorphic::FlatteningCleanupMode::MainAsRoot
          }
      )
  );
  // Unrolling exposes constant array indices needed by scalarization.
  pm.addPass(mlir::createCanonicalizerPass());
  // Split PODs before arrays so arrays of PODs become parallel scalar arrays.
  pm.addPass(llzk::pod::createPodToScalarPass());
  pm.addPass(llzk::array::createArrayToScalarPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createPCLLoweringPass());
}

void registerConversionPassPipelines() {
  mlir::PassPipelineRegistration<>(
      "llzk-full-pcl-lowering",
      "Normalize loops, flatten templates, and scalarize aggregates before lowering to PCL, "
      "preserving component boundaries",
      buildFullPCLLoweringPipeline
  );
}

} // namespace pcl
