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

  /// Return whether the value semantics of `type` can be preserved by either reusing the same SSA
  /// value or materializing independent mutable storage.
  ///
  /// Returning `false` means that reuse would introduce observable aliasing and this interface
  /// cannot currently construct the required independent copy.
  virtual bool canMaterializeValueCopy(mlir::Type type) const = 0;

  /// Preserve the value-copy semantics of `source` at the builder's insertion point.
  ///
  /// Implementations may return `source` only when its type has no independently mutable payload.
  /// Otherwise they must construct independent storage or return failure.
  virtual mlir::FailureOr<mlir::Value>
  materializeValueCopy(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source) const = 0;
};

/// Return whether `type` has supported value-copy semantics. A successful copy may reuse the same
/// SSA value only for immutable values, identity-bearing handles, or aggregates without mutable
/// payload.
bool canMaterializeValueCopy(mlir::Type type);

/// Return `source` for immutable scalar values, identity-bearing handles, and aggregates without
/// mutable payload, or materialize recursively independent storage for a supported mutable
/// aggregate. Return failure rather than aliasing `source` when an independent copy is required but
/// unsupported. The caller must set the desired insertion point first.
mlir::FailureOr<mlir::Value>
materializeValueCopy(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value source);

} // namespace llzk
