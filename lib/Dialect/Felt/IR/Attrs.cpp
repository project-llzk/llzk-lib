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

LogicalResult FeltConstAttr::verify(
    function_ref<InFlightDiagnostic()> errFn,
    APInt, // NOLINT(performance-unnecessary-value-param)
    StringAttr fieldName
) {
  return fieldName ? Field::verifyFieldDefined(
                         fieldName.getValue(), wrapNonNullableInFlightDiagnostic(errFn)
                     )
                   : success();
}

FeltType FeltConstAttr::getType() const {
  return FeltType::get(this->getContext(), this->getFieldName());
}

FeltConstAttr::operator APInt() const { return getValue(); }

FeltConstAttr FeltConstAttr::get(MLIRContext *context, APInt value, FeltType type) {
  // TODO: this attr should be updated to store the FeltType directly, but for now we can just
  // reconstruct it from the field name as needed.
  return FeltConstAttr::get(context, value, type.hasField() ? type.getFieldName() : StringAttr());
}

} // namespace llzk::felt
