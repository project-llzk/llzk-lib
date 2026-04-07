//===-- Felt.cpp - Felt dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include "llzk-c/Dialect/Felt.h"
#include "llzk-c/Dialect/LLZK.h"

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;

// Include the generated CAPI
#include "llzk/Dialect/Felt/IR/Attrs.capi.cpp.inc"
#include "llzk/Dialect/Felt/IR/Ops.capi.cpp.inc"
#include "llzk/Dialect/Felt/IR/Types.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt, FeltDialect)

MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetInField(MlirContext ctx, int64_t value, MlirStringRef fieldName) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value), unwrap(fieldName)));
}

MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetUnspecified(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value)));
}

MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value, MlirType type) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value), unwrap_cast<FeltType>(type)));
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetWithBitsInField(
    MlirContext ctx, unsigned numBits, int64_t value, MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value), unwrap(fieldName)));
}

MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetWithBitsUnspecified(MlirContext ctx, unsigned numBits, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value)));
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromString(
    MlirContext ctx, unsigned numBits, MlirStringRef str, MlirType type
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str), unwrap_cast<FeltType>(type)));
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromStringInField(
    MlirContext ctx, unsigned numBits, MlirStringRef str, MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str), unwrap(fieldName)));
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromStringUnspecified(
    MlirContext ctx, unsigned numBits, MlirStringRef str
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str)));
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext ctx, unsigned numBits, const uint64_t *parts, intptr_t nParts, MlirType type
) {
  return wrap(
      FeltConstAttr::get(unwrap(ctx), numBits, ArrayRef(parts, nParts), unwrap_cast<FeltType>(type))
  );
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromPartsInField(
    MlirContext ctx, unsigned numBits, const uint64_t *parts, intptr_t nParts,
    MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, ArrayRef(parts, nParts), unwrap(fieldName)));
}

MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromPartsUnspecified(
    MlirContext ctx, unsigned numBits, const uint64_t *parts, intptr_t nParts
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, ArrayRef(parts, nParts)));
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromString(
    MlirContext ctx, MlirIdentifier fieldName, unsigned numBits, MlirStringRef primeStr
) {
  return wrap(
      FieldSpecAttr::get(unwrap(ctx), unwrap(fieldName), APInt(numBits, unwrap(primeStr), 10))
  );
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromParts(
    MlirContext ctx, MlirIdentifier fieldName, unsigned numBits, const uint64_t *parts,
    intptr_t nParts
) {
  return wrap(
      FieldSpecAttr::get(unwrap(ctx), unwrap(fieldName), APInt(numBits, ArrayRef(parts, nParts)))
  );
}

MlirType llzkFelt_FeltTypeGetUnspecified(MlirContext ctx) {
  return wrap(FeltType::get(unwrap(ctx)));
}

MlirType llzkFelt_FeltTypeGetFromRef(MlirContext ctx, MlirStringRef fieldName) {
  return wrap(FeltType::get(unwrap(ctx), unwrap(fieldName)));
}
