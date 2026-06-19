//===-- Ops.cpp - Boolean operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Ops.h"

#include "llzk/Dialect/Bool/IR/Utils.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Support/LLVM.h>

#include <cassert>

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

inline static bool eval(FeltCmpPredicate pred, const llvm::APInt &lval, const llvm::APInt &rval) {
  switch (pred) {
  case FeltCmpPredicate::EQ:
    return lval == rval;
  case FeltCmpPredicate::NE:
    return lval != rval;
  case FeltCmpPredicate::LT:
    return lval.ult(rval);
  case FeltCmpPredicate::LE:
    return lval.ule(rval);
  case FeltCmpPredicate::GT:
    return lval.ugt(rval);
  case FeltCmpPredicate::GE:
    return lval.uge(rval);
  }
  llvm_unreachable("invalid FeltCmpPredicate");
}

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
  return makeBoolAttr(getContext(), eval(getPredicate(), lval, rval));
}

//===------------------------------------------------------------------===//
// Quantifier ops common impl
//===------------------------------------------------------------------===//

namespace {
/// Verifies a quantifier operation.
///
/// The region of the operation must have only 1 argument and its type must match
/// the element type of the sort.
template <typename Op> LogicalResult verifyQuantOp(Op op) {
  auto *block = op.getBody();
  if (!block || block->getNumArguments() != 1) {
    return op->emitOpError() << "must have one block argument";
  }
  auto argType = block->getArgument(0).getType();
  auto eltType = getQuantifierOpDomainIterType(op.getSort().getType());
  if (argType != eltType) {
    return op->emitOpError() << "expects element type " << argType << " but sort has element type "
                             << eltType;
  }

  auto termOp = block->getTerminator();
  if (!llvm::dyn_cast_if_present<YieldOp>(termOp)) {
    return op->emitOpError() << "expects 'bool.yield' terminator op";
  }
  return success();
}

/// Parses a quantifier operation.
///
/// The grammar is the same for both `forall` and `exists`:
///
/// ```
/// bool.(forall|exists) %elt (`:` type(%elt))? `in` $sort `:` type($sort) $region (`attributes`
/// attr-dict)?
/// ```
ParseResult parseQuantOp(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::Argument arg;
  if (parser.parseArgument(arg)) {
    return failure();
  }
  assert(!arg.type);
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseType(arg.type)) {
      return failure();
    }
  }
  if (parser.parseKeyword("in")) {
    return failure();
  }
  OpAsmParser::UnresolvedOperand sortOperand;
  array::ArrayType sortType;
  if (parser.parseOperand(sortOperand)) {
    return failure();
  }
  if (parser.parseColonType(sortType)) {
    return failure();
  }
  if (parser.resolveOperand(sortOperand, sortType, result.operands)) {
    return failure();
  }

  if (!arg.type) {
    arg.type = getQuantifierOpDomainIterType(sortType);
    assert(arg.type && "argument type must be inferred from the array element type");
  }

  auto *body = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(
          *body, {arg},
          /*enableNameShadowing=*/false
      )) {
    return failure();
  }

  if (body->empty()) {
    return parser.emitError(loc, "expected non-empty body");
  }
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return failure();
  }

  result.types = {parser.getBuilder().getI1Type()};
  return success();
}

/// Prints a quantifier operation.
template <typename Op> void printQuantOp(OpAsmPrinter &p, Op op) {
  p << ' ';
  p.printRegionArgument(op.getBody()->getArgument(0));
  p << " in ";
  p.printOperand(op.getSort());
  p << " : " << op.getSort().getType();
  p << ' ';
  p.printRegion(op.getRegion(), /*printEntryBlockArgs=*/false);
  p << ' ';
  p.printOptionalAttrDictWithKeyword(op->getAttrs());
}
} // namespace

//===------------------------------------------------------------------===//
// ForAllOp
//===------------------------------------------------------------------===//

LogicalResult ForAllOp::verify() { return verifyQuantOp(*this); }

ParseResult ForAllOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseQuantOp(parser, result);
}

void ForAllOp::print(OpAsmPrinter &p) { printQuantOp(p, *this); }

//===------------------------------------------------------------------===//
// ExistsOp
//===------------------------------------------------------------------===//

LogicalResult ExistsOp::verify() { return verifyQuantOp(*this); }

ParseResult ExistsOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseQuantOp(parser, result);
}

void ExistsOp::print(OpAsmPrinter &p) { printQuantOp(p, *this); }

} // namespace llzk::boolean

/// Extracts the type used for a quantifier op block argument.
///
/// If the array has only one dimension, returns the element type.
/// Otherwise, returns an array type with the first dimension removed.
Type llzk::boolean::getQuantifierOpDomainIterType(array::ArrayType arr) {
  if (arr.getDimensionSizes().size() == 1) {
    return arr.getElementType();
  }

  return array::ArrayType::get(arr.getElementType(), arr.getDimensionSizes().drop_front());
}
