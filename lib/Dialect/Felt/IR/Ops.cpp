//===-- Ops.cpp - Felt operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/Builders.h>

#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/SmallString.h>

// TableGen'd implementation files
#include "llzk/Dialect/Felt/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Felt/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk;

namespace llzk::felt {

//===------------------------------------------------------------------===//
// Constant folding helpers
//===------------------------------------------------------------------===//

namespace {

struct BinaryFoldData {
  DynamicAPInt lhsVal, rhsVal;
  StringRef fieldName;
  const Field *field;
};

struct UnaryFoldData {
  DynamicAPInt val;
  StringRef fieldName;
  const Field *field;
};

/// Returns fold inputs for a binary felt op, or nullopt if folding should not
/// proceed.  Folding is skipped when:
///   - either operand constant attribute is absent (non-constant operand), or
///   - either field name is unspecified (null StringAttr), or
///   - the two field names differ.
static std::optional<BinaryFoldData> tryGetBinaryFoldData(Attribute lhsAttr, Attribute rhsAttr) {
  auto lhs = llvm::dyn_cast_or_null<FeltConstAttr>(lhsAttr);
  auto rhs = llvm::dyn_cast_or_null<FeltConstAttr>(rhsAttr);
  if (!lhs || !rhs) {
    return std::nullopt;
  }

  StringAttr lhsFieldName = lhs.getFieldName();
  StringAttr rhsFieldName = rhs.getFieldName();
  if (!lhsFieldName || !rhsFieldName || lhsFieldName != rhsFieldName) {
    return std::nullopt;
  }

  auto fieldRes = Field::tryGetField(lhsFieldName.getValue());
  if (failed(fieldRes)) {
    return std::nullopt;
  }

  return BinaryFoldData {
      toDynamicAPInt(lhs.getValue()), toDynamicAPInt(rhs.getValue()), lhsFieldName.getValue(),
      &fieldRes.value().get()
  };
}

/// Same guard logic for unary felt ops.
static std::optional<UnaryFoldData> tryGetUnaryFoldData(Attribute operandAttr) {
  auto operand = llvm::dyn_cast_or_null<FeltConstAttr>(operandAttr);
  if (!operand) {
    return std::nullopt;
  }

  StringAttr fieldNameAttr = operand.getFieldName();
  if (!fieldNameAttr) {
    return std::nullopt;
  }

  auto fieldRes = Field::tryGetField(fieldNameAttr.getValue());
  if (failed(fieldRes)) {
    return std::nullopt;
  }

  return UnaryFoldData {
      toDynamicAPInt(operand.getValue()), fieldNameAttr.getValue(), &fieldRes.value().get()
  };
}

/// Builds a FeltConstAttr carrying the reduced result value.
static FeltConstAttr buildFoldResult(
    MLIRContext *ctx, const DynamicAPInt &val, const Field &field, StringRef fieldName
) {
  return FeltConstAttr::get(ctx, toAPInt(val, field.bitWidth()), fieldName);
}

} // namespace

//===------------------------------------------------------------------===//
// FeltConstantOp
//===------------------------------------------------------------------===//

void FeltConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf;
  llvm::raw_svector_ostream(buf) << "felt_const_";
  getValue().getValue().toStringUnsigned(buf);
  setNameFn(getResult(), buf);
}

OpFoldResult FeltConstantOp::fold(FeltConstantOp::FoldAdaptor) { return getValueAttr(); }

LogicalResult FeltConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> /*loc*/, Adaptor adaptor,
    SmallVectorImpl<Type> &inferred
) {
  inferred.resize(1);
  auto value = adaptor.getValue(); // FeltConstAttr
  inferred[0] = value ? value.getType() : FeltType::get(context, StringAttr());
  return success();
}

bool FeltConstantOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) { return l == r; }

//===------------------------------------------------------------------===//
// Binary op folds
//===------------------------------------------------------------------===//

OpFoldResult AddFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal + data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult SubFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal - data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult MulFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal * data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult PowFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), modExp(data->lhsVal, data->rhsVal, data->field->prime()), *data->field,
      data->fieldName
  );
}

OpFoldResult DivFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data || data->rhsVal == DynamicAPInt(0)) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal * data->field->inv(data->rhsVal)),
      *data->field, data->fieldName
  );
}

OpFoldResult UnsignedIntDivFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data || data->rhsVal == DynamicAPInt(0)) {
    return {};
  }
  // Both values are non-negative field elements; standard integer division
  // gives the correct unsigned quotient, already in [0, lhs] < prime.
  return buildFoldResult(getContext(), data->lhsVal / data->rhsVal, *data->field, data->fieldName);
}

OpFoldResult SignedIntDivFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  DynamicAPInt sRhs = data->field->toSigned(data->rhsVal);
  if (sRhs == DynamicAPInt(0)) {
    return {};
  }
  DynamicAPInt sLhs = data->field->toSigned(data->lhsVal);
  // DynamicAPInt / truncates toward zero (same as C++ signed int division).
  return buildFoldResult(
      getContext(), data->field->reduce(sLhs / sRhs), *data->field, data->fieldName
  );
}

OpFoldResult UnsignedModFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data || data->rhsVal == DynamicAPInt(0)) {
    return {};
  }
  // Both non-negative, so % gives the correct unsigned remainder in [0, rhs) < prime.
  return buildFoldResult(getContext(), data->lhsVal % data->rhsVal, *data->field, data->fieldName);
}

OpFoldResult SignedModFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  DynamicAPInt sRhs = data->field->toSigned(data->rhsVal);
  if (sRhs == DynamicAPInt(0)) {
    return {};
  }
  DynamicAPInt sLhs = data->field->toSigned(data->lhsVal);
  return buildFoldResult(
      getContext(), data->field->reduce(sLhs % sRhs), *data->field, data->fieldName
  );
}

OpFoldResult AndFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal & data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult OrFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal | data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult XorFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal ^ data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult ShlFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(data->lhsVal << data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult ShrFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  // Right-shifting a non-negative value always yields a value in [0, lhs] < prime;
  // no modular reduction required.
  return buildFoldResult(getContext(), data->lhsVal >> data->rhsVal, *data->field, data->fieldName);
}

//===------------------------------------------------------------------===//
// Unary op folds
//===------------------------------------------------------------------===//

OpFoldResult NegFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetUnaryFoldData(adaptor.getOperand());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), data->field->reduce(-data->val), *data->field, data->fieldName
  );
}

OpFoldResult InvFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetUnaryFoldData(adaptor.getOperand());
  if (!data || data->val == DynamicAPInt(0)) {
    return {};
  }
  return buildFoldResult(getContext(), data->field->inv(data->val), *data->field, data->fieldName);
}

OpFoldResult NotFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetUnaryFoldData(adaptor.getOperand());
  if (!data) {
    return {};
  }
  // One's complement at field.bitWidth() bits: maxMask = 2^bitWidth - 1,
  // result = reduce(maxMask ^ val).  The operator<< here is llzk::operator<<
  // on DynamicAPInt (defined in DynamicAPIntHelper.h).
  DynamicAPInt maxMask =
      (DynamicAPInt(1) << DynamicAPInt(data->field->bitWidth())) - DynamicAPInt(1);
  return buildFoldResult(
      getContext(), data->field->reduce(maxMask ^ data->val), *data->field, data->fieldName
  );
}

} // namespace llzk::felt
