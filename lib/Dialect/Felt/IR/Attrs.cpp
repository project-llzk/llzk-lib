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

StringAttr FeltConstAttr::getFieldName() const {
  auto ft = getType();
  return ft ? ft.getFieldName() : StringAttr();
}

FailureOr<FeltConstAttr> FeltConstAttr::materializeAs(FeltType expectedType) const {
  FeltType valueType = getType();
  if (valueType.hasField() && expectedType.hasField() && valueType != expectedType) {
    return failure();
  }
  return FeltConstAttr::get(
      getContext(), getValue(), expectedType.hasField() ? expectedType : valueType
  );
}

} // namespace llzk::felt
