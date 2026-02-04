//===-- OpTraits.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/OpTraits.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/IR/Operation.h>

using namespace mlir;

namespace llzk::function {

namespace {

auto parentFuncDefOpHasAttr = [](Operation *op, auto attrFn) -> bool {
  if (FuncDefOp f = op->getParentOfType<FuncDefOp>()) {
    return (f.*attrFn)();
  }
  return false;
};

} // namespace

LogicalResult verifyConstraintGenTraitImpl(Operation *op) {
  if (!parentFuncDefOpHasAttr(op, &FuncDefOp::hasAllowConstraintAttr)) {
    return op->emitOpError() << "only valid within a '" << FuncDefOp::getOperationName()
                             << "' with '" << AllowConstraintAttr::name << "' attribute";
  }
  return success();
}

LogicalResult verifyWitnessGenTraitImpl(Operation *op) {
  if (!parentFuncDefOpHasAttr(op, &FuncDefOp::hasAllowWitnessAttr)) {
    return op->emitOpError() << "only valid within a '" << FuncDefOp::getOperationName()
                             << "' with '" << AllowWitnessAttr::name << "' attribute";
  }
  return success();
}

LogicalResult verifyNotFieldNativeTraitImpl(Operation *op) {
  if (!parentFuncDefOpHasAttr(op, &FuncDefOp::hasAllowNonNativeFieldOpsAttr)) {
    return op->emitOpError() << "only valid within a '" << FuncDefOp::getOperationName()
                             << "' with '" << AllowNonNativeFieldOpsAttr::name << "' attribute";
  }
  return success();
}

} // namespace llzk::function
