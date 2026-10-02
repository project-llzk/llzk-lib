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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;

namespace {

/// Export the optional sidecar only after the binary R1CS stream was produced.
LogicalResult exportLayoutMap(ModuleOp module, StringRef selectedCircuit, StringRef layoutMapFile) {
  if (layoutMapFile.empty()) {
    return success();
  }

  std::string buffer;
  llvm::raw_string_ostream layout(buffer);
  if (failed(r1cs::exportLLZKLayoutMap(module, layout, selectedCircuit))) {
    return failure();
  }
  layout.flush();

  std::error_code error;
  auto layoutFile =
      std::make_unique<llvm::ToolOutputFile>(layoutMapFile, error, llvm::sys::fs::OF_None);
  if (error) {
    return module.emitError() << "could not open layout map '" << layoutMapFile
                              << "': " << error.message();
  }
  layoutFile->os() << buffer;
  layoutFile->os().flush();
  if (layoutFile->os().has_error()) {
    return module.emitError() << "could not write layout map '" << layoutMapFile << "'";
  }
  layoutFile->keep();
  return success();
}

/// Serialize the binary circuit and its optional layout using the selected options.
LogicalResult exportBinaryAndLayoutMap(
    ModuleOp module, llvm::raw_ostream &output, StringRef prime, StringRef circuitName,
    StringRef layoutMapFile
) {
  if (failed(r1cs::exportR1CSBinary(module, output, prime, circuitName))) {
    return failure();
  }
  return exportLayoutMap(module, circuitName, layoutMapFile);
}

/// Evaluate and lower LLZK, then serialize the same module without a text round-trip.
LogicalResult lowerAndExportR1CS(
    Operation *op, llvm::raw_ostream &output, StringRef prime, StringRef circuitName,
    StringRef layoutMapFile
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
  return exportBinaryAndLayoutMap(module, output, prime, circuitName, layoutMapFile);
}

} // namespace

void r1cs::registerR1CSTranslation() {
  static llvm::cl::OptionCategory r1csTranslationOptions("R1CS translation options");

  static llvm::cl::opt<std::string> prime(
      "r1cs-prime",
      llvm::cl::desc("Prime modulus as a base-10 integer (default: infer the unique LLZK field)"),
      llvm::cl::init(""), llvm::cl::cat(r1csTranslationOptions)
  );

  static llvm::cl::opt<std::string> circuitName(
      "r1cs-circuit-name",
      llvm::cl::desc("Circuit symbol to export when the module contains multiple circuits"),
      llvm::cl::init(""), llvm::cl::cat(r1csTranslationOptions)
  );

  static llvm::cl::opt<std::string> layoutMapFile(
      "llzk-layout-map",
      llvm::cl::desc("Write the LLZK signal layout and R1CS wire relation to this file"),
      llvm::cl::init(""), llvm::cl::cat(r1csTranslationOptions)
  );

  TranslateFromMLIRRegistration direct(
      "llzk-to-r1cs", "evaluate and lower LLZK directly to binary R1CS in memory",
      [](Operation *op, llvm::raw_ostream &output) {
    return lowerAndExportR1CS(op, output, prime, circuitName, layoutMapFile);
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
    return exportBinaryAndLayoutMap(moduleOp, output, prime, circuitName, layoutMapFile);
  }, [](DialectRegistry &registry) {
    llzk::registerDialects(registry);
    registry.insert<R1CSDialect>();
  }
  );
}
