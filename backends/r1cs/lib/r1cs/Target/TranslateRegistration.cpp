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

#include "llzk/Dialect/DialectRegistration.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Tools/mlir-translate/Translation.h>

#include <llvm/Support/CommandLine.h>

using namespace mlir;

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
