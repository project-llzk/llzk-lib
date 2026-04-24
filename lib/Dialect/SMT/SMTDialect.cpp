//===- SMTDialect.cpp - SMT dialect implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/SMT/IR/SMTDialect.h"

#include "llzk/Dialect/SMT/IR/SMTAttributes.h"
#include "llzk/Dialect/SMT/IR/SMTOps.h"
#include "llzk/Dialect/SMT/IR/SMTTypes.h"

using namespace mlir;
using namespace llzk::smt;

void SMTDialect::initialize() {
  registerAttributes();
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/SMT/IR/SMT.cpp.inc"
      >();
}

Operation *
SMTDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type, Location loc) {
  // BitVectorType constants can materialize into smt.bv.constant
  if (auto bvType = dyn_cast<BitVectorType>(type)) {
    if (auto attrValue = dyn_cast<BitVectorAttr>(value)) {
      assert(bvType == attrValue.getType() && "attribute and desired result types have to match");
      return builder.create<BVConstantOp>(loc, attrValue);
    }
  }

  // BoolType constants can materialize into smt.constant
  if (auto boolType = dyn_cast<BoolType>(type)) {
    if (auto attrValue = dyn_cast<BoolAttr>(value)) {
      return builder.create<BoolConstantOp>(loc, attrValue);
    }
  }

  return nullptr;
}

#include "llzk/Dialect/SMT/IR/SMTDialect.cpp.inc"
#include "llzk/Dialect/SMT/IR/SMTEnums.cpp.inc"
