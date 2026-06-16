//===-- LLZKTransformationPassPipelines.h -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/Parsers.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassOptions.h>

namespace llzk {

struct FullPolyLoweringOptions : public mlir::PassPipelineOptions<FullPolyLoweringOptions> {
  Option<unsigned> maxDegree {
      *this, "max-degree", llvm::cl::desc("Maximum polynomial degree (must be ≥ 2)"),
      llvm::cl::init(2)
  };
};

struct FullStructInliningOptions : public mlir::PassPipelineOptions<FullStructInliningOptions> {

  using FlatteningPipelinePassOptions = NestedPassOptions<
      static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::polymorphic::createFlatteningPass)>;
  using InlineStructsPipelinePassOptions = NestedPassOptions<
      static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::component::createInlineStructsPass)>;

  Option<FlatteningPipelinePassOptions> flattening {
      *this, "flattening", llvm::cl::desc("options for the flattening pass used in this pipeline"),
      llvm::cl::init(FlatteningPipelinePassOptions {})
  };
  Option<bool> arrayToScalar {
      *this, "array-to-scalar",
      llvm::cl::desc("whether to run the array-to-scalar pass in this pipeline"),
      llvm::cl::init(true)
  };
  Option<bool> podToScalar {
      *this, "pod-to-scalar",
      llvm::cl::desc("whether to run the pod-to-scalar pass in this pipeline"), llvm::cl::init(true)
  };
  Option<InlineStructsPipelinePassOptions> inlining {
      *this, "inlining", llvm::cl::desc("options for the inlining pass used in this pipeline"),
      llvm::cl::init(InlineStructsPipelinePassOptions {})
  };
};

void buildRemoveUnnecessaryOpsPipeline(mlir::OpPassManager &);

void buildRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &);

void buildFullPolyLoweringPipeline(mlir::OpPassManager &, const FullPolyLoweringOptions &);

void buildProductProgramPipeline(mlir::OpPassManager &);

void buildFullStructInliningPipeline(mlir::OpPassManager &, const FullStructInliningOptions &);

void registerTransformationPassPipelines();

} // namespace llzk
