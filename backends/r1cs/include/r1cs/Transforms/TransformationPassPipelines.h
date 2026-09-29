//===-- TransformationPassPipelines.h ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

namespace r1cs {

/// Select the input contract explicitly when assembling an R1CS pipeline.
enum class R1CSLoweringMode { Legacy, Direct };

/// Options for the existing full R1CS pipeline; direct storage lowering is the default.
struct FullR1CSLoweringOptions : public mlir::PassPipelineOptions<FullR1CSLoweringOptions> {
  Option<bool> legacy {
      *this, "legacy", llvm::cl::desc("Use legacy flattened-input lowering"), llvm::cl::init(false)
  };
};

/// Build a flat pipeline for either flattened legacy IR or evaluated storage IR.
void buildFullR1CSLoweringPipeline(
    mlir::OpPassManager &, R1CSLoweringMode mode = R1CSLoweringMode::Direct
);

void registerTransformationPassPipelines();

} // namespace r1cs
