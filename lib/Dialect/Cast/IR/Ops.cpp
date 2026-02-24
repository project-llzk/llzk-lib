//===-- Ops.cpp - Cast operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"

#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Cast/IR/Ops.cpp.inc"

namespace llzk::cast {

bool IntToFeltOp::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(llvm::zip_equal(lhs, rhs), [](auto pair) {
    auto [lhsType, rhsType] = pair;
    auto lhsFeltType = mlir::dyn_cast<llzk::felt::FeltType>(lhsType);
    auto rhsFeltType = mlir::dyn_cast<llzk::felt::FeltType>(rhsType);

    // If both types are felts but NOT structurally equal then check if the types are valid
    // with the additional consideration that lhs is allowed to NOT have
    // a declared field.
    if (lhsFeltType && rhsFeltType && lhsFeltType != rhsFeltType) {
      // If we reached this point we know that the felts are not equal and that only the lhs is
      // allowed to not have a declared field. Thus, rhs must have a declared field. If lhs has a
      // declared field, then, since they are not structurally equal, it must be a different field
      // than lhs. With all that, the types are compatible if lhs does not have a field, so we can
      // simply return that.
      return !lhsFeltType.hasField();
    }

    // Any other case gets handled by standard equality.
    return lhsType == rhsType;
  });
}
} // namespace llzk::cast
