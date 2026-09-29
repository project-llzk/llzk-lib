//===-- Attrs.cpp - Felt Attr method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"

#include "llzk/Util/DynamicAPIntHelper.h"

using namespace mlir;

namespace llzk::felt {

StringAttr FeltConstAttr::getFieldName() const {
  auto ft = getType();
  return ft ? ft.getFieldName() : StringAttr();
}

llvm::APInt FeltConstAttr::getReducedValue() const {
  auto type = getType();
  if (!type || !type.hasField()) {
    return getRawValue();
  }
  const Field &field = type.getField();
  return toAPInt(field.reduce(getRawValue()), field.bitWidth());
}

} // namespace llzk::felt
