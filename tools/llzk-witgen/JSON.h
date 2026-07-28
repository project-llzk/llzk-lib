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
enum class SerializationMode : std::uint8_t {
  PublicOutputsOnly,
  AllSignals,
};

/// One structured JSON mismatch between expected and actual witgen output.
struct JSONMismatch {
  std::string path;
  std::string message;
};

/// Parse one JSON value into the tool's runtime representation.
llvm::Expected<WitnessVal> parseJSONValue(
    const llvm::json::Value *json, mlir::Type type, const llzk::Field &field,
    mlir::Operation *origin
);

/// Serialize one runtime value into the user-facing JSON output format.
llvm::Expected<llvm::json::Value> serializeJSONValue(
    const WitnessVal &value, mlir::Type type, mlir::SymbolTableCollection &tables,
    mlir::Operation *origin, SerializationMode mode = SerializationMode::PublicOutputsOnly
);

/// Serialize named input values into a JSON object.
llvm::Expected<llvm::json::Object> buildInputsJSONObject(
    llvm::ArrayRef<InputBinding> bindings, llvm::ArrayRef<WitnessVal> values,
    mlir::SymbolTableCollection &tables, mlir::Operation *origin
);

/// Extract one nested runtime leaf by path.
llvm::Expected<WitnessVal> extractValueAtPath(
    const WitnessVal &root, mlir::Type rootType, llvm::ArrayRef<std::string> path,
    mlir::SymbolTableCollection &tables, mlir::Operation *origin
);

/// Compare two JSON values structurally and append any mismatches to `out`.
void diffJSON(
    const llvm::json::Value &expected, const llvm::json::Value &actual,
    llvm::SmallVectorImpl<JSONMismatch> &out, llvm::StringRef path = "$"
);

/// Render one human-readable mismatch report.
void printJSONMismatches(llvm::raw_ostream &os, llvm::ArrayRef<JSONMismatch> mismatches);

} // namespace llzk::witgen
