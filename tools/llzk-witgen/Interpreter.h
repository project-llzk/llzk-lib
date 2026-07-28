//===-- Interpreter.h - llzk-witgen compute interpreter ---------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ValueModel.h"

#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallVector.h>

#include <random>

namespace llzk::witgen {

/// Execute one flattened LLZK function body over runtime values.
class FunctionInterpreter {
public:
  /// Build an interpreter for one module and field configuration.
  FunctionInterpreter(
      mlir::ModuleOp moduleOp, mlir::SymbolTableCollection &tables, const llzk::Field &field,
      UninitializedBehavior uninitializedBehavior, std::mt19937_64 rng
  );

  /// Run a function with concrete arguments and return its result values.
  llvm::Expected<llvm::SmallVector<WitnessVal>>
  run(llzk::function::FuncDefOp funcOp, mlir::ArrayRef<WitnessVal> args);

private:
  mlir::ModuleOp moduleOp;
  mlir::SymbolTableCollection &tables;
  const llzk::Field &field;
  UninitializedBehavior uninitializedBehavior = UninitializedBehavior::Zero;
  std::mt19937_64 rng;
};

} // namespace llzk::witgen
