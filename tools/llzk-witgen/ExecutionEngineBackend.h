//===-- ExecutionEngineBackend.h - llzk-witgen JIT backend ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "WitgenDriver.h"

#include "llzk/Util/Field.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

namespace llzk::witgen {

/// Execute witness generation through MLIR lowering and `ExecutionEngine`.
llvm::Expected<llvm::json::Value> runWithExecutionEngine(
    mlir::ModuleOp moduleOp, mlir::SymbolTableCollection &tables, const llzk::Field &field,
    const llvm::json::Value &input, const WitgenOptions &options
);

} // namespace llzk::witgen
