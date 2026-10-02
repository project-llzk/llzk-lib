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
    llvm::function_ref<InFlightDiagnostic()> emitError, StringAttr, llzk::DynamicAPIntValue prime
) {
  if (static_cast<const llvm::DynamicAPInt &>(prime) < 2) {
    return emitError() << "field modulus must be at least 2";
  }
  return success();
}

StringAttr FeltConstAttr::getFieldName() const {
  auto ft = getType();
  return ft ? ft.getFieldName() : StringAttr();
}

} // namespace llzk::felt
