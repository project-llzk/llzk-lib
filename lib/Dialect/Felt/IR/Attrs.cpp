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
  FeltType ft = getType();
  return ft ? ft.getFieldName() : StringAttr();
}

std::optional<llvm::DynamicAPInt> FeltConstAttr::getReducedValue() const {
  FeltType type = getType();
  if (!type || !type.hasField()) {
    return std::nullopt;
  }
  const Field &field = type.getField();
  return field.reduce(getRawValue());
}

llvm::DynamicAPInt FeltConstAttr::getReducedValueOrRaw() const {
  if (auto reduced = getReducedValue()) {
    return *reduced;
  }
  return toDynamicAPInt(getRawValue());
}

} // namespace llzk::felt
