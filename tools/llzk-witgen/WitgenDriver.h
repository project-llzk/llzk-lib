//===-- WitgenDriver.h - llzk-witgen driver entrypoints ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ValueModel.h"

#include "llzk/Util/Field.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

#include <cstdint>
#include <optional>
#include <random>

namespace llzk::witgen {

/// Select the execution backend used by `llzk-witgen`.
enum class Backend {
  Interpreter,
  ExecutionEngine,
};

/// Select the JSON scope emitted by `llzk-witgen`.
enum class OutputScope {
  Public,
  FullWitness,
};

/// Configure one `llzk-witgen` execution.
struct WitgenOptions {
  Backend backend = Backend::Interpreter;
  OutputScope outputScope = OutputScope::Public;
  UninitializedBehavior uninitializedBehavior = UninitializedBehavior::Zero;
  std::optional<uint64_t> randomSeed;
  bool inlineIncludes = true;
  bool dumpJITCore = false;
  bool dumpJITLLVM = false;
};

/// Drive witness generation for the concrete `llzk.main` instance.
class Interpreter {
public:
  /// Build a driver for one parsed module and validated field.
  Interpreter(
      mlir::ModuleOp moduleOp, mlir::SymbolTableCollection &tables, const llzk::Field &field,
      UninitializedBehavior uninitializedBehavior, std::mt19937_64 rng
  );

  /// Select which witness JSON scope this interpreter emits.
  void setOutputScope(OutputScope newOutputScope) { outputScope = newOutputScope; }

  /// Execute the main `compute()` function using JSON inputs.
  llvm::Expected<llvm::json::Value> runMainFromJSON(const llvm::json::Value &input);

private:
  mlir::ModuleOp moduleOp;
  mlir::SymbolTableCollection &tables;
  const llzk::Field &field;
  OutputScope outputScope = OutputScope::Public;
  UninitializedBehavior uninitializedBehavior = UninitializedBehavior::Zero;
  std::mt19937_64 rng;
};

/// Run the full llzk-witgen pipeline on a parsed module.
llvm::Expected<llvm::json::Value>
runWitgen(mlir::ModuleOp moduleOp, const llvm::json::Value &input, const WitgenOptions &options);

} // namespace llzk::witgen
