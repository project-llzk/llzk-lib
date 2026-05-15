//===-- JSON.h - llzk-witgen JSON conversion helpers ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "ValueModel.h"
#include "WitnessSelection.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

namespace llzk::witgen {

/// Select how struct values are filtered during JSON serialization.
enum class SerializationMode {
  PublicOutputsOnly,
  AllSignals,
};

/// Parse one JSON value into the tool's runtime representation.
llvm::Expected<Value> parseJSONValue(
    const llvm::json::Value *json, mlir::Type type, const llzk::Field &field,
    mlir::Operation *origin
);

/// Serialize one runtime value into the user-facing JSON output format.
llvm::Expected<llvm::json::Value> serializeJSONValue(
    const Value &value, mlir::Type type, mlir::SymbolTableCollection &tables,
    mlir::Operation *origin,
    SerializationMode mode = SerializationMode::PublicOutputsOnly
);

/// Serialize named input values into a JSON object.
llvm::Expected<llvm::json::Object> buildInputsJSONObject(
    llvm::ArrayRef<InputBinding> bindings, llvm::ArrayRef<Value> values,
    mlir::SymbolTableCollection &tables, mlir::Operation *origin
);

/// Extract one nested runtime leaf by path.
llvm::Expected<Value> extractValueAtPath(
    const Value &root, mlir::Type rootType, llvm::ArrayRef<std::string> path,
    mlir::SymbolTableCollection &tables, mlir::Operation *origin
);

} // namespace llzk::witgen
