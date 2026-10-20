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

#include <llvm/ADT/Twine.h>

namespace llzk::polymorphic::detail {
/// Emit the common diagnostic prefix for constraint expansion and storage helpers.
inline mlir::LogicalResult constraintError(mlir::Operation *site, const llvm::Twine &message) {
  return site->emitError("constraint flattening: ") << message;
}
} // namespace llzk::polymorphic::detail
