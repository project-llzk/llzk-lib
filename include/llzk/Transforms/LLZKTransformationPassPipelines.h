//===-- LLZKTransformationPassPipelines.h -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

namespace llzk {

struct FullPolyLoweringOptions : public mlir::PassPipelineOptions<FullPolyLoweringOptions> {
  Option<unsigned> maxDegree {
      *this, "max-degree", llvm::cl::desc("Maximum polynomial degree (must be ≥ 2)"),
      llvm::cl::init(2)
  };
};

void buildRemoveUnnecessaryOpsPipeline(mlir::OpPassManager &);

void buildRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &);

void buildFullPolyLoweringPipeline(mlir::OpPassManager &, const FullPolyLoweringOptions &);

void buildProductProgramPipeline(mlir::OpPassManager &);

void registerTransformationPassPipelines();

} // namespace llzk
