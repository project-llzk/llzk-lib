//===-- ScalarEvaluation.h -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/Builders.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace llzk::polymorphic::detail {

/// Evaluate a scalar operation using caller-supplied operand constants.
/// Discovery supplies template/iteration bindings; flattening supplies SSA facts.
/// Cast fold hooks preserve their range checks and overflow semantics.
/// Fold hooks run on a disposable clone so evaluation never rewrites source IR.
inline mlir::LogicalResult evaluateScalarOperation(
    mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &results
) {
  // Assert casts fold only when their range checks are satisfied.
  if (op->getNumRegions() || (!mlir::isMemoryEffectFree(op) &&
                              !llvm::isa<llzk::cast::FeltToIndexOp, llzk::cast::IntToFeltOp>(op))) {
    return mlir::failure();
  }
  mlir::Operation *temporary = op->clone();
  auto status = temporary->fold(operands, results);
  temporary->destroy();
  return mlir::success(mlir::succeeded(status) && results.size() == op->getNumResults());
}

} // namespace llzk::polymorphic::detail
