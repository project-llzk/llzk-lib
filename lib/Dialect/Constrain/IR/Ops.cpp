//===-- Ops.cpp - Constrain operation implementations -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/ErrorHelper.h"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Constrain/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::array;

namespace llzk::constrain {

//===------------------------------------------------------------------===//
// EmitEqualityOp
//===------------------------------------------------------------------===//

LogicalResult EmitEqualityOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getLhs().getType(), getRhs().getType()}
  );
}

Type EmitEqualityOp::inferRHS(Type lhsType) { return lhsType; }

//===------------------------------------------------------------------===//
// EmitContainmentOp
//===------------------------------------------------------------------===//

LogicalResult EmitContainmentOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(
      tables, *this, ArrayRef<Type> {getLhs().getType(), getRhs().getType()}
  );
}

LogicalResult EmitContainmentOp::verify() {
  auto arrType = llvm::cast<ArrayType>(getLhs().getType()); // per the ODS definition
  OwningEmitErrorFn errFn = getEmitOpErrFn(this);

  if (failed(verifySubArrayOrElementType(errFn, arrType, getRhs().getType()))) {
    // error already reported
    return failure();
  }
  // The types are known to unify at this point; we can now check that the
  // array element type is a valid emit equal type.
  Type elemTy = arrType.getElementType();
  if (!isValidEmitEqType(elemTy)) {
    return errFn().append(
        "element type must be any LLZK type, excluding struct and string types, but got ", elemTy
    );
  }
  return success();
}

} // namespace llzk::constrain
