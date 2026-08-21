//===-- ProductSourceHelper.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Constants.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/StringRef.h>

#include <optional>

namespace llzk {

/// Return the product-source value stored directly on `op`, if present.
///
/// `compute` and `constrain` identify one source role. `fused` identifies an operation containing
/// both roles so later transforms do not treat it as compute-only or constrain-only. Product-source
/// helpers inspect only the attribute stored on the operation; they do not infer a role from nested
/// operations.
inline std::optional<llvm::StringRef> getProductSource(mlir::Operation *op) {
  if (mlir::StringAttr source = op->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
    return source.getValue();
  }
  return std::nullopt;
}

/// Return whether `op` stores the requested product-source value.
inline bool hasProductSource(mlir::Operation *op, llvm::StringRef source) {
  std::optional<llvm::StringRef> productSource = getProductSource(op);
  return productSource && *productSource == source;
}

/// Set the product-source value used to classify aligned and fused operations.
inline void setProductSource(mlir::Operation *op, llvm::StringRef source) {
  op->setAttr(PRODUCT_SOURCE, mlir::StringAttr::get(op->getContext(), source));
}

} // namespace llzk
