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

/// Typed nested options for the flattening pass when used inside the full
/// struct-inlining and full poly-lowering pipelines.
struct StructInliningFlatteningOptions
    : public mlir::PassPipelineOptions<StructInliningFlatteningOptions> {
  // Implementation note: these options should be kept in sync with the `FlatteningPass` ODS.

  Option<unsigned> iterationLimit {
      *this, "max-iter", llvm::cl::desc("maximum number of flattening iterations before giving up"),
      llvm::cl::init(1000)
  };

  Option<polymorphic::FlatteningCleanupMode> cleanupMode {
      *this, "cleanup",
      llvm::cl::desc(
          "cleanup mode for flattening in this pipeline. When left as `unspecified`, these "
          "pipelines use `main-as-root`. Overriding this is not recommended because the later "
          "`llzk-inline-structs` pass may crash if parameterized templates survive flattening."
      ),
      llvm::cl::init(polymorphic::FlatteningCleanupMode::Unspecified)
  };

  polymorphic::FlatteningPassOptions createPassOptions() const {
    return polymorphic::FlatteningPassOptions {
        .iterationLimit = iterationLimit, .cleanupMode = cleanupMode
    };
  }
};

/// Pure C++ configuration for the full struct inlining pipeline.
struct FullStructInliningConfig {
  polymorphic::FlatteningPassOptions flattening;
  bool arrayToScalar = true;
  bool podToScalar = true;
  component::InlineStructsPassOptions inlining;
};

/// CLI Option configuration for the full struct inlining pipeline.
struct FullStructInliningOptions : public mlir::PassPipelineOptions<FullStructInliningOptions> {

  using FlatteningOptions = NestedPipelineOptions<StructInliningFlatteningOptions>;

  using InliningOptions = NestedPassOptions<
      static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::component::createInlineStructsPass)>;

  Option<FlatteningOptions> flattening {
      *this, "flattening",
      llvm::cl::desc(
          "options for the flattening pass used in this pipeline; this pipeline defaults "
          "flattening pass cleanup to `main-as-root`"
      ),
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

  using PolyLoweringOptions =
      NestedPassOptions<static_cast<std::unique_ptr<mlir::Pass> (*)()>(&llzk::createPolyLoweringPass
      )>;

  Option<StructInliningOptions> structInlining {
      *this, "flatten-inline",
      llvm::cl::desc(
          "options for the struct flattening and inlining pipeline used before polynomial "
          "lowering; this pipeline defaults flattening cleanup to `main-as-root`"
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
