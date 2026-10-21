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

/// Evaluate a pure scalar operation using caller-supplied operand constants.
/// Discovery supplies template/iteration bindings; flattening supplies SSA facts.
/// Canonicalization-only casts share the conversion semantics of their rewrites.
/// Fold hooks run on a disposable clone so evaluation never rewrites source IR.
inline mlir::LogicalResult evaluateScalarOperation(
    mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands,
    llvm::SmallVectorImpl<mlir::OpFoldResult> &results
) {
  if (op->getNumRegions() || !mlir::isMemoryEffectFree(op)) {
    return mlir::failure();
  }
  if (auto conversion = llvm::dyn_cast<llzk::cast::IntToFeltOp>(op)) {
    auto constant = llvm::dyn_cast_or_null<mlir::IntegerAttr>(operands.front());
    if (!constant) {
      return mlir::failure();
    }
    results.push_back(
        felt::FeltConstAttr::get(
            op->getContext(), integerAttrToDynamicAPInt(constant), conversion.getType()
        )
    );
    return mlir::success();
  }
  if (llvm::isa<llzk::cast::FeltToIndexOp>(op)) {
    auto constant = llvm::dyn_cast_or_null<felt::FeltConstAttr>(operands.front());
    if (!constant) {
      return mlir::failure();
    }
    auto reduced = constant.getReducedValue();
    if (!reduced) {
      return mlir::failure();
    }
    auto converted = checkedToInt64(*reduced);
    if (!converted) {
      llvm::consumeError(converted.takeError());
      return mlir::failure();
    }
    if (*converted < 0) {
      return mlir::failure();
    }
    results.push_back(mlir::Builder(op->getContext()).getIndexAttr(*converted));
    return mlir::success();
  }
  mlir::Operation *temporary = op->clone();
  auto status = temporary->fold(operands, results);
  temporary->destroy();
  return mlir::success(mlir::succeeded(status) && results.size() == op->getNumResults());
}

} // namespace llzk::polymorphic::detail
