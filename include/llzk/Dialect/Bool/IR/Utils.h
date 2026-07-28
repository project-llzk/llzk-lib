//===-- Dialect.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Types.h"

namespace llzk::boolean {

/// Extracts the type used for a quantifier op block argument.
///
/// If the array has only one dimension, returns the element type.
/// Otherwise, returns an array type with the first dimension removed.
inline mlir::Type getQuantifierOpDomainIterType(llzk::array::ArrayType arr) {
  return arr.getSelectionType(1);
}

} // namespace llzk::boolean
