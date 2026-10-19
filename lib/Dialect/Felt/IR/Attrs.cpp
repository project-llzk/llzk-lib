//===-- Attrs.cpp - Felt Attr method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"

using namespace mlir;

namespace llzk::felt {

LogicalResult FieldSpecAttr::verify(
    llvm::function_ref<InFlightDiagnostic()> emitError, StringAttr,
    // TableGen requires this parameter to be passed by value.
    llzk::DynamicAPIntValue primeWrapper // NOLINT(performance-unnecessary-value-param)
) {
  const llvm::DynamicAPInt &prime = primeWrapper; // implict unwrap
  if (prime < 2) {
    return emitError() << "field modulus must be at least 2";
  }
  return success();
}

StringAttr FeltConstAttr::getFieldName() const {
  FeltType ft = getType();
  return ft ? ft.getFieldName() : StringAttr();
}

std::optional<llvm::DynamicAPInt> FeltConstAttr::getReducedValue() const {
  FeltType type = getType();
  if (!type || !type.hasField()) {
    return std::nullopt;
  }
  return type.getField().reduce(getRawValue());
}

llvm::DynamicAPInt FeltConstAttr::getReducedValueOrRaw() const {
  if (auto reduced = getReducedValue()) {
    return *reduced;
  }
  return getRawValue();
}

} // namespace llzk::felt
