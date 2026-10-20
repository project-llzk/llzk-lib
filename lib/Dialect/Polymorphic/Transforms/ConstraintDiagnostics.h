//===-- ConstraintDiagnostics.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>

namespace llzk::polymorphic::detail {
/// Identify expansion diagnostics separately from IR verifier errors.
inline constexpr llvm::StringLiteral CONSTRAINT_DIAGNOSTIC_PREFIX = "llzk-flatten-constraints: ";

/// Emit the common diagnostic prefix for constraint expansion and storage helpers.
inline mlir::LogicalResult constraintError(mlir::Operation *site, const llvm::Twine &message) {
  return site->emitError(CONSTRAINT_DIAGNOSTIC_PREFIX) << message;
}
} // namespace llzk::polymorphic::detail
