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

#include <gmp.h>
#include <type_traits>

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

/// Constant binary operands and their shared field.
template <typename Int> struct BinaryFoldData {
  Int lhsVal, rhsVal;
  StringRef fieldName;
  const Field *field;
};

/// Constant unary operand and its field.
template <typename Int> struct UnaryFoldData {
  Int val;
  StringRef fieldName;
  const Field *field;
};

/// Returns fold inputs for a binary felt op, or nullopt if folding should not
/// proceed.  Folding is skipped when:
///   - either operand constant attribute is absent (non-constant operand), or
///   - either field name is unspecified (null StringAttr), or
///   - the two field names differ.
template <typename Int = DynamicAPInt>
static std::optional<BinaryFoldData<Int>>
tryGetBinaryFoldData(Attribute lhsAttr, Attribute rhsAttr) {
  static_assert(
      std::is_same_v<Int, APInt> || std::is_same_v<Int, DynamicAPInt>,
      "fold inputs must use APInt or DynamicAPInt"
  );
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

  if constexpr (std::is_same_v<Int, APInt>) {
    return BinaryFoldData<Int> {
        lhs.getValue(), rhs.getValue(), lhsFieldName.getValue(), &fieldRes.value().get()
    };
  } else {
    return BinaryFoldData<Int> {
        toDynamicAPInt(lhs.getValue()), toDynamicAPInt(rhs.getValue()), lhsFieldName.getValue(),
        &fieldRes.value().get()
    };
  }
}

/// Same guard logic for unary felt ops.
template <typename Int = DynamicAPInt>
static std::optional<UnaryFoldData<Int>> tryGetUnaryFoldData(Attribute operandAttr) {
  static_assert(
      std::is_same_v<Int, APInt> || std::is_same_v<Int, DynamicAPInt>,
      "fold inputs must use APInt or DynamicAPInt"
  );
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

  if constexpr (std::is_same_v<Int, APInt>) {
    return UnaryFoldData<Int> {
        operand.getValue(), fieldNameAttr.getValue(), &fieldRes.value().get()
    };
  } else {
    return UnaryFoldData<Int> {
        toDynamicAPInt(operand.getValue()), fieldNameAttr.getValue(), &fieldRes.value().get()
    };
  }
}

/// Builds a FeltConstAttr carrying the reduced result value.
static FeltConstAttr buildFoldResult(
    MLIRContext *ctx, const DynamicAPInt &val, const Field &field, StringRef fieldName
) {
  return FeltConstAttr::get(ctx, toAPInt(val, field.bitWidth()), fieldName);
}

/// Reduce an exact unsigned intermediate only when it reaches the modulus.
/// Returned attributes still have canonical field values and a clear sign bit.
static APInt reduceUnsigned(APInt value, const Field &field) {
  unsigned width = std::max(value.getBitWidth(), field.primeAPInt().getBitWidth());
  value = value.zext(width);
  APInt prime = field.primeAPInt().zext(width);
  if (value.uge(prime)) {
    value = value.urem(prime);
  }
  return value.zextOrTrunc(field.bitWidth() + 1);
}

/// Add without losing carry bits, widening only if the original width overflows.
static APInt exactAdd(const APInt &lhs, const APInt &rhs) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  APInt a = lhs.zext(width), b = rhs.zext(width);
  bool overflow;
  APInt result = a.uadd_ov(b, overflow);
  return overflow ? a.zext(width + 1) + b.zext(width + 1) : result;
}

/// Multiply exactly, widening only if the original width overflows.
static APInt exactMultiply(const APInt &lhs, const APInt &rhs) {
  unsigned width = std::max(lhs.getBitWidth(), rhs.getBitWidth());
  APInt a = lhs.zext(width), b = rhs.zext(width);
  bool overflow;
  APInt result = a.umul_ov(b, overflow);
  if (!overflow) {
    return result;
  }
  width = std::max(width, lhs.getActiveBits() + rhs.getActiveBits());
  return lhs.zext(width) * rhs.zext(width);
}

/// Unsigned modular subtraction, including negative integer differences.
static APInt subtractModulo(const APInt &lhs, const APInt &rhs, const Field &field) {
  APInt a = reduceUnsigned(lhs, field), b = reduceUnsigned(rhs, field);
  return a.uge(b) ? a - b : field.primeAPInt().zext(a.getBitWidth()) - (b - a);
}

/// Exponentiation by squaring with exact products and reduced intermediates.
static APInt powerModulo(APInt base, const APInt &exponent, const Field &field) {
  base = reduceUnsigned(base, field);
  APInt result(base.getBitWidth(), 1);
  unsigned bits = exponent.getActiveBits();
  for (unsigned i = 0; i < bits; ++i) {
    if (exponent[i]) {
      result = reduceUnsigned(exactMultiply(result, base), field);
    }
    if (i + 1 < bits) {
      base = reduceUnsigned(exactMultiply(base, base), field);
    }
  }
  return result;
}

/// Compute an inverse with GMP's extended GCD, transferring unsigned machine
/// words directly. A non-invertible operand must remain unfolded.
static std::optional<APInt> inverseModulo(const APInt &value, const Field &field) {
  mpz_t operand, modulus, inverse;
  mpz_inits(operand, modulus, inverse, nullptr);
  const APInt &prime = field.primeAPInt();
  mpz_import(operand, value.getNumWords(), -1, sizeof(uint64_t), 0, 0, value.getRawData());
  mpz_import(modulus, prime.getNumWords(), -1, sizeof(uint64_t), 0, 0, prime.getRawData());
  std::optional<APInt> result;
  if (mpz_invert(inverse, operand, modulus)) {
    unsigned width = field.bitWidth() + 1;
    SmallVector<uint64_t> words((width + 63) / 64, 0);
    mpz_export(words.data(), nullptr, -1, sizeof(uint64_t), 0, 0, inverse);
    result = APInt(width, words);
  }
  mpz_clears(operand, modulus, inverse, nullptr);
  return result;
}

/// Construct a canonical constant without any DynamicAPInt round trip.
static FeltConstAttr
buildFoldResult(MLIRContext *ctx, const APInt &val, const Field &field, StringRef fieldName) {
  return FeltConstAttr::get(ctx, reduceUnsigned(val, field), fieldName);
}

} // namespace

//===------------------------------------------------------------------===//
// FeltConstantOp
//===------------------------------------------------------------------===//

void FeltConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buf;
  llvm::raw_svector_ostream(buf) << "felt_const_";
  getValueAPInt().toStringUnsigned(buf);
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
  auto data = tryGetBinaryFoldData<APInt>(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), exactAdd(data->lhsVal, data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult SubFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData<APInt>(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), subtractModulo(data->lhsVal, data->rhsVal, *data->field), *data->field,
      data->fieldName
  );
}

OpFoldResult MulFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData<APInt>(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), exactMultiply(data->lhsVal, data->rhsVal), *data->field, data->fieldName
  );
}

OpFoldResult PowFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData<APInt>(adaptor.getLhs(), adaptor.getRhs());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), powerModulo(data->lhsVal, data->rhsVal, *data->field), *data->field,
      data->fieldName
  );
}

OpFoldResult DivFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData<APInt>(adaptor.getLhs(), adaptor.getRhs());
  if (!data || data->rhsVal == 0) {
    return {};
  }
  auto inverse = inverseModulo(data->rhsVal, *data->field);
  if (!inverse) {
    return {};
  }
  return buildFoldResult(
      getContext(), exactMultiply(data->lhsVal, *inverse), *data->field, data->fieldName
  );
}

OpFoldResult UnsignedIntDivFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data || data->rhsVal == 0) {
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
  const Field *field = data->field;
  DynamicAPInt rhs = data->rhsVal;
  if (rhs == 0 || rhs == field->prime()) {
    return {};
  }
  DynamicAPInt sRhs = field->toSigned(rhs);
  DynamicAPInt sLhs = field->toSigned(data->lhsVal);
  // DynamicAPInt / truncates toward zero (same as C++ signed int division).
  return buildFoldResult(getContext(), field->reduce(sLhs / sRhs), *field, data->fieldName);
}

OpFoldResult UnsignedModFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetBinaryFoldData(adaptor.getLhs(), adaptor.getRhs());
  if (!data || data->rhsVal == 0) {
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
  const Field *field = data->field;
  DynamicAPInt rhs = data->rhsVal;
  if (rhs == 0 || rhs == field->prime()) {
    return {};
  }
  DynamicAPInt sRhs = field->toSigned(rhs);
  DynamicAPInt sLhs = field->toSigned(data->lhsVal);
  return buildFoldResult(getContext(), field->reduce(sLhs % sRhs), *field, data->fieldName);
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
  // Any shift `amount >= bitwidth` will yield zero.
  const Field *field = data->field;
  if (data->rhsVal >= DynamicAPInt(field->bitWidth())) {
    return buildFoldResult(getContext(), DynamicAPInt(0), *field, data->fieldName);
  }
  // Right-shifting a non-negative value always yields a value in [0, lhs] < prime;
  // no modular reduction required.
  return buildFoldResult(getContext(), data->lhsVal >> data->rhsVal, *field, data->fieldName);
}

//===------------------------------------------------------------------===//
// Unary op folds
//===------------------------------------------------------------------===//

OpFoldResult NegFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetUnaryFoldData<APInt>(adaptor.getOperand());
  if (!data) {
    return {};
  }
  return buildFoldResult(
      getContext(), subtractModulo(APInt(1, 0), data->val, *data->field), *data->field,
      data->fieldName
  );
}

OpFoldResult InvFeltOp::fold(FoldAdaptor adaptor) {
  auto data = tryGetUnaryFoldData<APInt>(adaptor.getOperand());
  if (!data || data->val == 0) {
    return {};
  }
  auto inverse = inverseModulo(data->val, *data->field);
  if (!inverse) {
    return {};
  }
  return buildFoldResult(getContext(), *inverse, *data->field, data->fieldName);
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
