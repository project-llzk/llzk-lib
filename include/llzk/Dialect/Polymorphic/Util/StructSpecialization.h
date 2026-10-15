//===-- StructSpecialization.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <optional>

namespace llzk::polymorphic {

/// Resolve concrete struct specializations referenced by poly.family metadata.
/// Specialization definitions, IDs, and origins must remain unchanged while this
/// resolver is in use. Unrelated malformed IDs are ignored; referenced IDs must
/// be unique and their poly.origin must match the source struct type.
class StructSpecializationResolver {
  /// Module scope used for specialization IDs and relative poly.origin paths.
  mlir::ModuleOp rootModule;
  /// Lazily built ID index; duplicate definitions are retained for diagnostics.
  std::optional<llvm::DenseMap<int64_t, llvm::SmallVector<component::StructDefOp, 1>>> definitions;

public:
  explicit StructSpecializationResolver(mlir::ModuleOp module) : rootModule(module) {}

  /// Resolve an ID and validate its source. Emit errors on site, identifying the
  /// caller using diagnosticContext (for example, "LLZK layout").
  mlir::FailureOr<component::StructDefOp> resolve(
      mlir::SymbolTableCollection &tables, component::StructType type, int64_t id,
      mlir::Operation *site, mlir::StringRef diagnosticContext
  );
};

} // namespace llzk::polymorphic
