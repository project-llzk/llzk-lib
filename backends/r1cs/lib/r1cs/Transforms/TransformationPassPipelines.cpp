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

void buildFullR1CSLoweringPipeline(OpPassManager &pm, const R1CSOutputConfig &outputs) {
  // 1. Polynomial degree lowering and cleanup
  llzk::FullPolyLoweringConfig config;
  config.polyLowering =
      llzk::PolyLoweringPassOptions {.markAuxiliarySignals = true, .maxDegree = 2};
  llzk::buildFullPolyLoweringPipeline(pm, config);

  // 2. Convert to R1CS
  pm.addPass(
      r1cs::createR1CSLoweringPass(
          r1cs::R1CSLoweringPassOptions {
              .outputLLZK = outputs.outputLLZK, .outputLayout = outputs.outputLayout
          }
      )
  );

  // 3. Run CSE to eliminate to_linear ops
  pm.addPass(mlir::createCSEPass());

  // Other passes that may be helpful to add in the future:
  // - llzk::createRemoveDeadValuesWorkaroundPass()
  // - mlir::createCanonicalizerPass()
  //    (was run via poly-lowering -> struct-inlining but again may be useful)
}

void buildConstraintFlatteningR1CSLoweringPipeline(
    OpPassManager &pm, const R1CSOutputConfig &outputs
) {
  llzk::buildConstraintFlatteningPipeline(pm);
  pm.addPass(
      llzk::createPolyLoweringPass(llzk::PolyLoweringPassOptions {.mainOnly = true, .maxDegree = 2})
  );
  pm.addPass(
      r1cs::createR1CSLoweringPass(
          r1cs::R1CSLoweringPassOptions {
              .outputLLZK = outputs.outputLLZK, .outputLayout = outputs.outputLayout
          }
      )
  );
  pm.addPass(mlir::createCSEPass());
}

namespace {

/// CLI filenames for the LLZK and CSV files emitted alongside R1CS IR.
struct R1CSOutputOptions : PassPipelineOptions<R1CSOutputOptions> {
  Option<std::string> outputLLZK {
      *this, "output-llzk",
      llvm::cl::desc("Write LLZK with auxiliary fields and compute assignments"), llvm::cl::init("")
  };
  Option<std::string> outputLayout {
      *this, "output-layout", llvm::cl::desc("Write the LLZK-to-R1CS layout CSV"),
      llvm::cl::init("")
  };
};

} // namespace

void registerTransformationPassPipelines() {
  PassPipelineRegistration<R1CSOutputOptions>(
      "llzk-constraint-flattening-r1cs-lowering",
      "Monomorphize and flatten main constraints, lower degree, and convert layout signals to R1CS",
      [](OpPassManager &pm, const R1CSOutputOptions &options) {
    buildConstraintFlatteningR1CSLoweringPipeline(pm, {options.outputLLZK, options.outputLayout});
  }
  );
  PassPipelineRegistration<R1CSOutputOptions>(
      "llzk-full-r1cs-lowering", "Lower polynomial constraints to r1cs",
      [](OpPassManager &pm, const R1CSOutputOptions &options) {
    buildFullR1CSLoweringPipeline(pm, {options.outputLLZK, options.outputLayout});
  }
  );
}

} // namespace r1cs
