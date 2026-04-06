//===-- Ops.cpp - Boolean operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinAttributes.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Bool/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::boolean {

//===------------------------------------------------------------------===//
// AssertOp
//===------------------------------------------------------------------===//

// This side effect models "program termination". Based on
// https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
void AssertOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects
) {
  effects.emplace_back(MemoryEffects::Write::get());
}

//===------------------------------------------------------------------===//
// Fold helpers
//===------------------------------------------------------------------===//

namespace {

/// Extract a bool value from an i1 IntegerAttr operand. Returns failure() if
/// the operand is not an IntegerAttr with i1 type.
static FailureOr<bool> getBoolValue(Attribute attr) {
  auto ia = llvm::dyn_cast_or_null<IntegerAttr>(attr);
  if (!ia || !ia.getType().isInteger(1)) {
    return failure();
  }
  return ia.getValue().getBoolValue();
}

/// Build an i1 IntegerAttr from a bool value.
static IntegerAttr makeBoolAttr(MLIRContext *ctx, bool val) {
  auto i1Ty = IntegerType::get(ctx, 1);
  return IntegerAttr::get(i1Ty, val ? 1 : 0);
}

} // namespace

//===------------------------------------------------------------------===//
// AndBoolOp
//===------------------------------------------------------------------===//

OpFoldResult AndBoolOp::fold(FoldAdaptor adaptor) {
  auto lhs = getBoolValue(adaptor.getLhs());
  auto rhs = getBoolValue(adaptor.getRhs());
  if (failed(lhs) || failed(rhs)) {
    return {};
  }
  return makeBoolAttr(getContext(), *lhs && *rhs);
}

//===------------------------------------------------------------------===//
// OrBoolOp
//===------------------------------------------------------------------===//

OpFoldResult OrBoolOp::fold(FoldAdaptor adaptor) {
  auto lhs = getBoolValue(adaptor.getLhs());
  auto rhs = getBoolValue(adaptor.getRhs());
  if (failed(lhs) || failed(rhs)) {
    return {};
  }
  return makeBoolAttr(getContext(), *lhs || *rhs);
}

//===------------------------------------------------------------------===//
// XorBoolOp
//===------------------------------------------------------------------===//

OpFoldResult XorBoolOp::fold(FoldAdaptor adaptor) {
  auto lhs = getBoolValue(adaptor.getLhs());
  auto rhs = getBoolValue(adaptor.getRhs());
  if (failed(lhs) || failed(rhs)) {
    return {};
  }
  return makeBoolAttr(getContext(), *lhs != *rhs);
}

//===------------------------------------------------------------------===//
// NotBoolOp
//===------------------------------------------------------------------===//

OpFoldResult NotBoolOp::fold(FoldAdaptor adaptor) {
  auto val = getBoolValue(adaptor.getOperand());
  if (failed(val)) {
    return {};
  }
  return makeBoolAttr(getContext(), !*val);
}

//===------------------------------------------------------------------===//
// CmpOp
//===------------------------------------------------------------------===//

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = llvm::dyn_cast_or_null<felt::FeltConstAttr>(adaptor.getLhs());
  auto rhsAttr = llvm::dyn_cast_or_null<felt::FeltConstAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr) {
    return {};
  }

  // Normalize to a common bit width for unsigned comparison.
  llvm::APInt lval = lhsAttr.getValue();
  llvm::APInt rval = rhsAttr.getValue();
  unsigned w = std::max(lval.getBitWidth(), rval.getBitWidth());
  if (lval.getBitWidth() < w) {
    lval = lval.zext(w);
  }
  if (rval.getBitWidth() < w) {
    rval = rval.zext(w);
  }

  bool result;
  switch (getPredicate()) {
  case FeltCmpPredicate::EQ:
    result = lval == rval;
    break;
  case FeltCmpPredicate::NE:
    result = lval != rval;
    break;
  case FeltCmpPredicate::LT:
    result = lval.ult(rval);
    break;
  case FeltCmpPredicate::LE:
    result = lval.ule(rval);
    break;
  case FeltCmpPredicate::GT:
    result = lval.ugt(rval);
    break;
  case FeltCmpPredicate::GE:
    result = lval.uge(rval);
    break;
  }
  return makeBoolAttr(getContext(), result);
}

} // namespace llzk::boolean
