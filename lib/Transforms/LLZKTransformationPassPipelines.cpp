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

#include "smt/Transforms/SMTPasses.h"

#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>

using namespace mlir;

namespace llzk {
struct FullPolyLoweringOptions : public PassPipelineOptions<FullPolyLoweringOptions> {
  Option<unsigned> maxDegree {
      *this, "max-degree", llvm::cl::desc("Maximum polynomial degree (must be ≥ 2)"),
      llvm::cl::init(2)
  };
};

struct VerifToSmtPipelineOptions : public PassPipelineOptions<VerifToSmtPipelineOptions> {
  Option<bool> cleanup {
      *this, "cleanup",
      llvm::cl::desc("Erase original LLZK symbol definitions after lowering to SMT helpers"),
      llvm::cl::init(true)
  };
};

struct VerifToSmtlibPipelineOptions : public PassPipelineOptions<VerifToSmtlibPipelineOptions> {
  Option<bool> cleanup {
      *this, "cleanup",
      llvm::cl::desc("Erase original LLZK symbol definitions before exporting SMTLIB"),
      llvm::cl::init(true)
  };
  Option<std::string> outputFilename {
      *this, "output-file",
      llvm::cl::desc("File to write the SMTLIB script to. Use '-' to write to stdout."),
      llvm::cl::init("-")
  };
  Option<std::string> entry {
      *this, "entry", llvm::cl::desc("Export only the specified func.func root symbol."),
      llvm::cl::init("")
  };
  Option<std::string> logic {
      *this, "logic",
      llvm::cl::desc("SMTLIB logic name to emit at the start of each exported script."),
      llvm::cl::init("ALL")
  };
};

static void addVerifToSmtPipeline(OpPassManager &pm, bool cleanup) {
  pm.addPass(llzk::createAggregateScalarizationPass());
  pm.addPass(llzk::createVerifToSmtPass(VerifToSmtPassOptions {.cleanup = cleanup}));
}

void addRemoveUnnecessaryOpsAndDefsPipeline(OpPassManager &pm) {
  pm.addPass(createRedundantReadAndWriteEliminationPass());
  pm.addPass(createRedundantOperationEliminationPass());
  pm.addPass(createUnusedDeclarationEliminationPass());
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
      "Lower already-flattened polynomial constraints to a given max degree, then remove "
      "unnecessary operations and definitions.",
      [](OpPassManager &pm, const FullPolyLoweringOptions &opts) {
    // 1. Degree lowering
    pm.addPass(createPolyLoweringPass(PolyLoweringPassOptions {.maxDegree = opts.maxDegree}));

    // 2. Cleanup
    addRemoveUnnecessaryOpsAndDefsPipeline(pm);
  }
  );

  PassPipelineRegistration<>(
      "llzk-product-program",
      "Convert @compute/@constrain functions to @product function and perform alignment",
      [](OpPassManager &pm) {
    pm.addPass(createComputeConstrainToProductPass());
    pm.addPass(createFuseProductLoopsPass());
  }
  );

  PassPipelineRegistration<VerifToSmtPipelineOptions>(
      "llzk-verif-to-smt",
      "Normalize array/pod aggregates and lower verif contracts to SMT helpers",
      [](OpPassManager &pm, const VerifToSmtPipelineOptions &opts) {
    addVerifToSmtPipeline(pm, opts.cleanup);
  }
  );

  PassPipelineRegistration<VerifToSmtlibPipelineOptions>(
      "llzk-verif-to-smtlib",
      "Normalize array/pod aggregates, lower verif contracts to SMT helpers, and export SMTLIB",
      [](OpPassManager &pm, const VerifToSmtlibPipelineOptions &opts) {
    addVerifToSmtPipeline(pm, opts.cleanup);
    pm.addPass(
        llzk::smt::createSMTDialectToSMTLIBPass(
            llzk::smt::SMTDialectToSMTLIBPassOptions {
                .outputFilename = opts.outputFilename, .entry = opts.entry, .logic = opts.logic
            }
        )
    );
  }
  );
}

} // namespace llzk
