//===-- ProductSourceHelper.h ----------------------------------*- C++ -*-===//
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

/// Return the compute/constrain source recorded on an aligned product operation, if present.
inline std::optional<llvm::StringRef> getProductSource(mlir::Operation *op) {
  if (mlir::StringAttr source = op->getAttrOfType<mlir::StringAttr>(PRODUCT_SOURCE)) {
    return source.getValue();
  }
  return std::nullopt;
}

/// Return whether an aligned product operation records the requested source.
inline bool hasProductSource(mlir::Operation *op, llvm::StringRef source) {
  std::optional<llvm::StringRef> productSource = getProductSource(op);
  return productSource && *productSource == source;
}

/// Record the compute/constrain source of an operation produced while aligning product functions.
inline void setProductSource(mlir::Operation *op, llvm::StringRef source) {
  op->setAttr(PRODUCT_SOURCE, mlir::StringAttr::get(op->getContext(), source));
}

} // namespace llzk
