//===-- LLZKTransformationPassPipelines.cpp ---------------------*- C++ -*-===//
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

#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <memory>
#include <string>

using namespace mlir;

namespace llzk {
namespace {

static std::string printOperationToString(Operation *op) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op->print(os);
  os.flush();
  return buffer;
}

class VerifAggregateScalarizationPass
    : public PassWrapper<VerifAggregateScalarizationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VerifAggregateScalarizationPass)

  StringRef getArgument() const final { return "llzk-verif-aggregate-scalarization"; }
  StringRef getDescription() const final {
    return "Repeatedly scalarize array/pod aggregates until the module stabilizes";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    constexpr unsigned kMaxIterations = 8;

    for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration) {
      std::string before = printOperationToString(module);

      OpPassManager pm(ModuleOp::getOperationName());
      pm.addPass(llzk::array::createArrayToScalarPass());
      pm.addPass(llzk::pod::createPodToScalarPass());
      pm.addPass(createCanonicalizerPass());
      if (failed(runPipeline(pm, module))) {
        signalPassFailure();
        return;
      }

      std::string after = printOperationToString(module);
      if (before == after) {
        return;
      }
    }

    module.emitError(
    ) << "verif aggregate scalarization did not reach a fixpoint within the iteration limit";
    signalPassFailure();
  }
};

} // namespace

struct FullPolyLoweringOptions : public PassPipelineOptions<FullPolyLoweringOptions> {
  Option<unsigned> maxDegree {
      *this, "max-degree", llvm::cl::desc("Maximum polynomial degree (must be ≥ 2)"),
      llvm::cl::init(2)
  };
};

void addRemoveUnnecessaryOpsAndDefsPipeline(OpPassManager &pm) {
  pm.addPass(llzk::createRedundantReadAndWriteEliminationPass());
  pm.addPass(llzk::createRedundantOperationEliminationPass());
  pm.addPass(llzk::createUnusedDeclarationEliminationPass());
}

std::unique_ptr<Pass> createVerifAggregateScalarizationPass() {
  return std::make_unique<VerifAggregateScalarizationPass>();
}

void registerTransformationPassPipelines() {
  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops",
      "Remove unnecessary operations, such as redundant reads or repeated constraints",
      [](OpPassManager &pm) {
    pm.addPass(createRedundantReadAndWriteEliminationPass());
    pm.addPass(createRedundantOperationEliminationPass());
  }
  );

  PassPipelineRegistration<>(
      "llzk-remove-unnecessary-ops-and-defs",
      "Remove unnecessary operations, member definitions, and struct definitions",
      [](OpPassManager &pm) { addRemoveUnnecessaryOpsAndDefsPipeline(pm); }
  );

  PassPipelineRegistration<FullPolyLoweringOptions>(
      "llzk-full-poly-lowering",
      "Lower all polynomial constraints to a given max degree, then remove unnecessary operations "
      "and definitions.",
      [](OpPassManager &pm, const FullPolyLoweringOptions &opts) {
    // 1. Degree lowering
    pm.addPass(llzk::createPolyLoweringPass(opts.maxDegree));

    // 2. Cleanup
    addRemoveUnnecessaryOpsAndDefsPipeline(pm);
  }
  );

  PassPipelineRegistration<>(
      "llzk-product-program",
      "Convert @compute/@constrain functions to @product function and perform alignment",
      [](OpPassManager &pm) {
    pm.addPass(llzk::createComputeConstrainToProductPass());
    pm.addPass(llzk::createFuseProductLoopsPass());
  }
  );

  PassPipelineRegistration<>(
      "llzk-verif-to-smt",
      "Normalize array/pod aggregates and lower verif contracts to SMT helpers",
      [](OpPassManager &pm) {
    pm.addPass(llzk::createVerifAggregateScalarizationPass());
    pm.addPass(llzk::createVerifToSmtPass());
  }
  );
}

} // namespace llzk
