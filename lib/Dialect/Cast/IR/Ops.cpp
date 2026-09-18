//===-- Ops.cpp - Cast operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Ops.h"

#include "llzk/Dialect/Cast/IR/Enums.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Cast/IR/Ops.cpp.inc"
using namespace mlir;

static inline ParseResult
parseOptionalOverflowSemantics(OpAsmParser &parser, llzk::cast::OverflowSemanticsAttr &overflow) {
  StringRef keyword;
  if (failed(parser.parseOptionalKeyword(&keyword))) {
    return success();
  }

  std::optional<llzk::cast::OverflowSemantics> semantics =
      llzk::cast::symbolizeOverflowSemantics(keyword);
  if (!semantics) {
    return parser.emitError(parser.getCurrentLocation()) << "expected overflow semantics keyword";
  }

  overflow = llzk::cast::OverflowSemanticsAttr::get(parser.getContext(), *semantics);
  return success();
}

static inline void
printOptionalOverflowSemantics(OpAsmPrinter &printer, llzk::cast::OverflowSemanticsAttr overflow) {
  if (!overflow || overflow.getValue() == llzk::cast::OverflowSemantics::ASSERT) {
    return;
  }

  printer << ' ' << stringifyOverflowSemantics(overflow.getValue());
}

namespace llzk::cast {

bool IntToFeltOp::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) {
  return lhs.size() == rhs.size() && llvm::all_of(llvm::zip_equal(lhs, rhs), [](auto pair) {
    auto [lhsType, rhsType] = pair;
    auto lhsFeltType = llvm::dyn_cast<llzk::felt::FeltType>(lhsType);
    auto rhsFeltType = llvm::dyn_cast<llzk::felt::FeltType>(rhsType);

    // If both types are felts but NOT structurally equal then check if the types are valid
    // with the additional consideration that lhs is allowed to NOT have
    // a declared field.
    if (lhsFeltType && rhsFeltType && lhsFeltType != rhsFeltType) {
      // If we reached this point we know that the felts are not equal and that only the lhs is
      // allowed to not have a declared field. Thus, rhs must have a declared field. If lhs has a
      // declared field, then, since they are not structurally equal, it must be a different field
      // than rhs. With all that, the types are compatible if lhs does not have a field, so we can
      // simply return that.
      return !lhsFeltType.hasField();
    }

    // Any other case gets handled by standard equality.
    return lhsType == rhsType;
  });
}

LogicalResult IntToFeltOp::canonicalize(IntToFeltOp op, ::mlir::PatternRewriter &rewriter) {
  // Instead of casting an arith.constant to felt, just generate a felt.const
  if (!op.getValue().getDefiningOp()) {
    return failure();
  }

  return llvm::TypeSwitch<Operation *, LogicalResult>(op.getValue().getDefiningOp())
      .Case<arith::ConstantIndexOp, arith::ConstantIntOp>([&rewriter, &op](auto constOp) {
    APInt value = toAPInt(constOp.value());
    felt::FeltType resultType = op.getType();

    // A field-less felt defers its field selection, so its overflow behavior
    // cannot be resolved while canonicalizing. Preserve the existing constant
    // representation in that case.
    if (!resultType.hasField()) {
      rewriter.replaceOpWithNewOp<felt::FeltConstantOp>(
          op, felt::FeltConstAttr::get(op->getContext(), value, resultType)
      );
      return success();
    }

    const Field &field = resultType.getField();
    DynamicAPInt signedValue = toSignedDynamicAPInt(value);
    DynamicAPInt result = signedValue;

    switch (op.getOverflow()) {
    case OverflowSemantics::ASSERT:
      if (signedValue < 0 || signedValue >= field.prime()) {
        return failure();
      }
      break;
    case OverflowSemantics::SATURATE:
      result = signedValue < 0 ? field.zero()
                               : (signedValue < field.maxVal() ? signedValue : field.maxVal());
      break;
    case OverflowSemantics::WRAP:
      result = field.reduce(signedValue);
      break;
    case OverflowSemantics::TRUNCATE:
      // Preserve exactly the target field bitwidth, without reducing modulo
      // the field prime.
      value = value.zextOrTrunc(field.bitWidth()).zext(field.bitWidth() + 1);
      rewriter.replaceOpWithNewOp<felt::FeltConstantOp>(
          op, felt::FeltConstAttr::get(op->getContext(), value, resultType)
      );
      return success();
    }

    rewriter.replaceOpWithNewOp<felt::FeltConstantOp>(
        op,
        felt::FeltConstAttr::get(op->getContext(), toAPInt(result, field.bitWidth()), resultType)
    );
    return success();
  }).Default([](auto) { return failure(); });
}

ParseResult
IntToFeltOp::parseOptionalOverflowSemantics(OpAsmParser &parser, OverflowSemanticsAttr &overflow) {
  return ::parseOptionalOverflowSemantics(parser, overflow);
}

void IntToFeltOp::printOptionalOverflowSemantics(
    OpAsmPrinter &printer, IntToFeltOp /*op*/, OverflowSemanticsAttr overflow
) {
  ::printOptionalOverflowSemantics(printer, overflow);
}

LogicalResult FeltToIndexOp::canonicalize(FeltToIndexOp op, ::mlir::PatternRewriter &rewriter) {
  // Instead of casting a felt.const to index, just generate an arith.constant
  if (auto constOp = op.getValue().getDefiningOp<felt::FeltConstantOp>()) {
    auto value = constOp.getValue().getValue();
    switch (op.getOverflow()) {
    case OverflowSemantics::ASSERT:
      // The sign check also protects programmatically constructed attributes
      // whose APInt width was not normalized by the textual IR parser.
      if (value.isNegative() || value.getActiveBits() > 63) {
        return failure();
      }
      rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, value.getSExtValue());
      return success();
    case OverflowSemantics::SATURATE:
      if (value.isNegative()) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, 0);
      } else if (value.getActiveBits() > 63) {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, INT64_MAX);
      } else {
        rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, value.getSExtValue());
      }
      return success();
    case OverflowSemantics::WRAP:
    case OverflowSemantics::TRUNCATE:
      rewriter.replaceOp(op, llzk::buildSafeIndexConstant(rewriter, op.getLoc(), value));
      return success();
    }
  }
  return failure();
}

ParseResult FeltToIndexOp::parseOptionalOverflowSemantics(
    OpAsmParser &parser, OverflowSemanticsAttr &overflow
) {
  return ::parseOptionalOverflowSemantics(parser, overflow);
}

void FeltToIndexOp::printOptionalOverflowSemantics(
    OpAsmPrinter &printer, FeltToIndexOp /*op*/, OverflowSemanticsAttr overflow
) {
  ::printOptionalOverflowSemantics(printer, overflow);
}

} // namespace llzk::cast
