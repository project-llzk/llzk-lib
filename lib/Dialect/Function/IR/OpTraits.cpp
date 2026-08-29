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
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/StringRef.h>

using namespace mlir;

namespace llzk::function {

namespace {

auto parentFuncDefOpHasAttr = [](Operation *op, auto attrFn) -> bool {
  if (FuncDefOp f = op->getParentOfType<FuncDefOp>()) {
    return (f.*attrFn)();
  }
  return false;
};

template <typename Attr, typename F>
LogicalResult verifyTraitIsPresentInFuncDefOp(Operation *op, F attrFn) {
  // These are allowed anywhere outside of FuncDefOp but only allowed inside a FuncDefOp
  // that is marked with the associated attribute.
  if (FuncDefOp f = op->getParentOfType<FuncDefOp>()) {
    if (!(f.*attrFn)()) {
      return op->emitOpError() << "cannot be used within a '" << FuncDefOp::getOperationName()
                               << "' without the '" << Attr::name << "' attribute";
    }
  }
  return success();
}

} // namespace

LogicalResult verifyConstraintGenTraitImpl(Operation *op) {
  if (parentFuncDefOpHasAttr(op, &FuncDefOp::hasAllowConstraintAttr)) {
    return success();
  }
  return op->emitOpError() << "only valid within a '" << FuncDefOp::getOperationName() << "' with '"
                           << AllowConstraintAttr::name << "' attribute";
}

LogicalResult verifyWitnessGenTraitImpl(Operation *op) {
  if (parentFuncDefOpHasAttr(op, &FuncDefOp::hasAllowWitnessAttr)) {
    return success();
  }
  return op->emitOpError() << "only valid within a '" << FuncDefOp::getOperationName() << "' with '"
                           << AllowWitnessAttr::name << "' attribute";
}

LogicalResult verifyNotFieldNativeTraitImpl(Operation *op) {
  return verifyTraitIsPresentInFuncDefOp<AllowNonNativeFieldOpsAttr>(
      op, &FuncDefOp::hasAllowNonNativeFieldOpsAttr
  );
}

LogicalResult
verifyVerificationTraitImpl(Operation *op, llvm::function_ref<LogicalResult()> check) {
  if (failed(
          verifyTraitIsPresentInFuncDefOp<AllowVerifOpsAttr>(op, &FuncDefOp::hasAllowVerifOpsAttr)
      )) {
    return failure();
  }
  if (FuncDefOp f = op->getParentOfType<FuncDefOp>()) {
    if (check) {
      return check();
    }
  }

  return success();
}

} // namespace llzk::function
