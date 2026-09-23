//===-- ValueCopy.h - Mutable value-copy materialization --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectInterface.h>
#include <mlir/Support/LLVM.h>

namespace llzk {

/// Dialect hook for materializing independent storage for mutable value types.
///
/// Implementations must recursively copy every mutable value child. Immutable scalar values and
/// identity-bearing handles are handled by the common entry points below and do not require a
/// dialect implementation.
class ValueCopyDialectInterface : public mlir::DialectInterface::Base<ValueCopyDialectInterface> {
public:
  using Base::Base;

  /// Return whether `type` and all of its mutable children can be copied explicitly.
  virtual bool canMaterializeValueCopy(mlir::Type type) const = 0;

  /// Create an independent value copy of `source` at the builder's insertion point.
  virtual mlir::FailureOr<mlir::Value>
  materializeValueCopy(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source) const = 0;
};

/// Return whether `type` has supported value-copy semantics.
bool canMaterializeValueCopy(mlir::Type type);

/// Return `source` for immutable scalar values and identity-bearing handles, or materialize
/// recursively independent storage for a supported mutable aggregate. The caller must set the
/// desired insertion point first.
mlir::FailureOr<mlir::Value>
materializeValueCopy(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source);

} // namespace llzk
