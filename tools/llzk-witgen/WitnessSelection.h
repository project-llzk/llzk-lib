//===-- WitnessSelection.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "WitgenDriver.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>

#include <string>

namespace llzk::witgen {

/// Describe one JSON-visible main input binding.
struct InputBinding {
  std::string name;
  mlir::Type type;
  unsigned index = 0;
};

/// Describe one selected witness output leaf.
struct OutputBinding {
  llvm::SmallVector<std::string> path;
  mlir::Type type;
};

/// Return `true` iff the member is considered a witness signal.
bool memberIsSignal(component::StructDefOp owner, component::MemberDefOp member);

/// Collect stable JSON bindings for the main compute inputs.
llvm::SmallVector<InputBinding> collectInputBindings(llzk::function::FuncDefOp computeFunc);

/// Collect the selected output bindings for the requested scope.
mlir::FailureOr<llvm::SmallVector<OutputBinding>> collectOutputBindings(
    component::StructDefOp mainDef, mlir::SymbolTableCollection &tables, mlir::Operation *origin,
    OutputScope scope
);

/// Assemble a nested JSON object from selected witness leaves.
llvm::json::Value buildSignalsJSONObject(
    llvm::ArrayRef<OutputBinding> bindings, llvm::ArrayRef<llvm::json::Value> serializedLeaves
);

} // namespace llzk::witgen
