//===-- TranslateRegistration.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Target/TranslateRegistration.h"

#include "r1cs/Dialect/IR/Dialect.h"
#include "r1cs/Target/R1CSBinary.h"
#include "r1cs/Transforms/TransformationPassPipelines.h"

#include "llzk/Dialect/DialectRegistration.h"
#include "llzk/Dialect/Polymorphic/Transforms/ConstraintEvaluation.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include <llvm/Support/CommandLine.h>

using namespace mlir;

namespace {

/// Evaluate and lower LLZK, then serialize the same module without a text round-trip.
LogicalResult lowerAndExportR1CS(
    Operation *op, llvm::raw_ostream &output, StringRef prime, StringRef circuitName
) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    return op->emitOpError() << "expected builtin.module as top level operation";
  }
  PassManager pm(module.getContext());
  if (!module->hasAttr(llzk::polymorphic::EVALUATED_MAIN_ATTR_NAME)) {
    pm.addPass(llzk::polymorphic::createDefinitionMonomorphizationPass());
    pm.addPass(llzk::polymorphic::createSymbolicConstraintEvaluationPass());
  }
  r1cs::buildFullR1CSLoweringPipeline(pm, r1cs::R1CSLoweringMode::Direct);
  if (failed(pm.run(module))) {
    return failure();
  }
  return r1cs::exportR1CSBinary(module, output, prime, circuitName);
}

} // namespace

void r1cs::registerR1CSTranslation() {
  static llvm::cl::OptionCategory r1csTranslationOptions("R1CS translation options");

  static llvm::cl::opt<std::string> prime(
      "r1cs-prime", llvm::cl::desc("Prime modulus as a base-10 integer"), llvm::cl::init(""),
      llvm::cl::cat(r1csTranslationOptions)
  );

  static llvm::cl::opt<std::string> circuitName(
      "r1cs-circuit-name",
      llvm::cl::desc("Circuit symbol to export when the module contains multiple circuits"),
      llvm::cl::init(""), llvm::cl::cat(r1csTranslationOptions)
  );

  TranslateFromMLIRRegistration direct(
      "llzk-to-r1cs", "evaluate and lower LLZK directly to binary R1CS in memory",
      [](Operation *op, llvm::raw_ostream &output) {
    return lowerAndExportR1CS(op, output, prime, circuitName);
  }, [](DialectRegistry &registry) {
    llzk::registerDialects(registry);
    registry.insert<R1CSDialect>();
  }
  );
  TranslateFromMLIRRegistration reg(
      "r1cs-to-binary", "translate R1CS IR to the binary .r1cs format",
      [](Operation *op, llvm::raw_ostream &output) -> LogicalResult {
    auto moduleOp = dyn_cast<ModuleOp>(op);
    if (!moduleOp) {
      return op->emitOpError() << "expected builtin.module as top level operation";
    }
    return exportR1CSBinary(moduleOp, output, prime, circuitName);
  }, [](DialectRegistry &registry) {
    llzk::registerDialects(registry);
    registry.insert<R1CSDialect>();
  }
  );
}
