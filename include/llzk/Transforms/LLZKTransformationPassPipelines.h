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

/// Pure C++ configuration for the full struct inlining pipeline.
struct FullStructInliningConfig {
  polymorphic::FlatteningPassOptions flattening;
  bool arrayToScalar = true;
  bool podToScalar = true;
  component::InlineStructsPassOptions inlining;
};

/// CLI Option configuration for the full struct inlining pipeline.
struct FullStructInliningOptions : public mlir::PassPipelineOptions<FullStructInliningOptions> {

  using FlatteningOptions = NestedPassOptions<
      static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::polymorphic::createFlatteningPass)>;
  using InliningOptions = NestedPassOptions<
      static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::component::createInlineStructsPass)>;

  Option<FlatteningOptions> flattening {
      *this, "flattening", llvm::cl::desc("options for the flattening pass used in this pipeline"),
      llvm::cl::init(FlatteningOptions {})
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
  Option<InliningOptions> inlining {
      *this, "inlining", llvm::cl::desc("options for the inlining pass used in this pipeline"),
      llvm::cl::init(InliningOptions {})
  };
};

/// Pure C++ configuration for the full polynomial lowering pipeline.
struct FullPolyLoweringConfig {
  FullStructInliningConfig structInlining;
  PolyLoweringPassOptions polyLowering;
};

/// CLI Option configuration for the full polynomial lowering pipeline.
struct FullPolyLoweringOptions : public mlir::PassPipelineOptions<FullPolyLoweringOptions> {

  using StructInliningOptions = NestedPipelineOptions<FullStructInliningOptions>;

  using PolyLoweringOptions = NestedPassOptions<
      static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::createPolyLoweringPass)>;

  Option<StructInliningOptions> structInlining {
      *this, "flatten-inline",
      llvm::cl::desc(
          "options for the struct flattening and inlining pipeline used before polynomial lowering"
      ),
      llvm::cl::init(StructInliningOptions {})
  };
  Option<PolyLoweringOptions> polyLowering {
      *this, "lowering",
      llvm::cl::desc("options for the polynomial lowering pass used in this pipeline"),
      llvm::cl::init(PolyLoweringOptions {})
  };
};

void buildRemoveUnnecessaryOpsPipeline(mlir::OpPassManager &);

void buildRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &);

void buildFullPolyLoweringPipeline(mlir::OpPassManager &, const FullPolyLoweringConfig &);

void buildProductProgramPipeline(mlir::OpPassManager &);

void buildFullStructInliningPipeline(mlir::OpPassManager &, const FullStructInliningConfig &);

void registerTransformationPassPipelines();

} // namespace llzk
