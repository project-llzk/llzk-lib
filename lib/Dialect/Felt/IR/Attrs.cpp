//===-- Attrs.cpp - Felt Attr method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"

namespace llzk::felt {

llvm::LogicalResult FeltConstAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> errFn, llvm::APInt, mlir::StringAttr fieldName
) {
  return fieldName ? Field::verifyFieldDefined(fieldName.getValue(), errFn) : mlir::success();
}

mlir::Type FeltConstAttr::getType() const {
  return FeltType::get(this->getContext(), this->getFieldName());
}

FeltConstAttr::operator ::llvm::APInt() const { return getValue(); }

} // namespace llzk::felt
