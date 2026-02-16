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

using namespace mlir;

namespace llzk::felt {

LogicalResult
FeltConstAttr::verify(function_ref<InFlightDiagnostic()> errFn, APInt, StringAttr fieldName) {
  return fieldName ? Field::verifyFieldDefined(
                         fieldName.getValue(), wrapNonNullableInFlightDiagnostic(errFn)
                     )
                   : success();
}

Type FeltConstAttr::getType() const {
  return FeltType::get(this->getContext(), this->getFieldName());
}

FeltConstAttr::operator APInt() const { return getValue(); }

} // namespace llzk::felt
