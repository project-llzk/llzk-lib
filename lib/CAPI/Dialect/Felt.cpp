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

MlirAttribute llzkFelt_FeltConstAttrGetUnspecified(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value)));
}

MlirAttribute llzkFelt_FeltConstAttrGetWithBits(
    MlirContext ctx, unsigned numBits, int64_t value, MlirIdentifier fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value), unwrap(fieldName)));
}

MlirAttribute
llzkFelt_FeltConstAttrGetWithBitsUnspecified(MlirContext ctx, unsigned numBits, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromString(
    MlirContext ctx, unsigned numBits, MlirStringRef str, MlirIdentifier fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str), unwrap(fieldName)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromStringUnspecified(
    MlirContext ctx, unsigned numBits, MlirStringRef str
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts,
    MlirIdentifier fieldName
) {
  return wrap(
      FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts), unwrap(fieldName))
  );
}

MlirAttribute llzkFelt_FeltConstAttrGetFromPartsUnspecified(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
) {
  return wrap(FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts)));
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromString(
    MlirContext context, MlirIdentifier fieldName, unsigned numBits, MlirStringRef primeStr
) {
  return wrap(
      FieldSpecAttr::get(unwrap(context), unwrap(fieldName), APInt(numBits, unwrap(primeStr), 10))
  );
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromParts(
    MlirContext context, MlirIdentifier fieldName, unsigned numBits, const uint64_t *parts,
    intptr_t nParts
) {
  return wrap(
      FieldSpecAttr::get(
          unwrap(context), unwrap(fieldName), APInt(numBits, ArrayRef(parts, nParts))
      )
  );
}

MlirType llzkFelt_FeltTypeGetUnspecified(MlirContext ctx) {
  return wrap(FeltType::get(unwrap(ctx)));
}
