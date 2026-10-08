//===-- Felt.cpp - Felt dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Felt.h"

#include "llzk-c/Dialect/LLZK.h"

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

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

MlirAttribute llzkFelt_FeltConstAttrGetFromInt64(MlirContext ctx, int64_t value, MlirType type) {
  return wrap(
      FeltConstAttr::get(unwrap(ctx), llvm::DynamicAPInt(value), unwrap_cast<FeltType>(type))
  );
}

MlirAttribute
llzkFelt_FeltConstAttrGetFromInt64InField(MlirContext ctx, int64_t value, MlirStringRef fieldName) {
  return wrap(FeltConstAttr::get(unwrap(ctx), llvm::DynamicAPInt(value), unwrap(fieldName)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromInt64Unspecified(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), llvm::DynamicAPInt(value)));
}

MlirAttribute
llzkFelt_FeltConstAttrGetFromString(MlirContext ctx, MlirStringRef str, MlirType type) {
  return llzkFelt_FeltConstAttrGet(ctx, str, type);
}

MlirAttribute llzkFelt_FeltConstAttrGetFromStringInField(
    MlirContext ctx, MlirStringRef str, MlirStringRef fieldName
) {
  return llzkFelt_FeltConstAttrGetFromString(ctx, str, llzkFelt_FeltTypeGetFromRef(ctx, fieldName));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromStringUnspecified(MlirContext ctx, MlirStringRef str) {
  return llzkFelt_FeltConstAttrGetFromString(ctx, str, llzkFelt_FeltTypeGetUnspecified(ctx));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext ctx, const uint64_t *parts, intptr_t nParts, MlirType type
) {
  assert(nParts >= 0 && "part count must be non-negative");
  assert((parts || nParts == 0) && "non-empty parts must not be null");
  return wrap(
      FeltConstAttr::get(unwrap(ctx), llvm::ArrayRef(parts, nParts), unwrap_cast<FeltType>(type))
  );
}

MlirAttribute llzkFelt_FeltConstAttrGetFromPartsInField(
    MlirContext ctx, const uint64_t *parts, intptr_t nParts, MlirStringRef fieldName
) {
  return llzkFelt_FeltConstAttrGetFromParts(
      ctx, parts, nParts, llzkFelt_FeltTypeGetFromRef(ctx, fieldName)
  );
}

MlirAttribute llzkFelt_FeltConstAttrGetFromPartsUnspecified(
    MlirContext ctx, const uint64_t *parts, intptr_t nParts
) {
  return llzkFelt_FeltConstAttrGetFromParts(
      ctx, parts, nParts, llzkFelt_FeltTypeGetUnspecified(ctx)
  );
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromString(
    MlirContext ctx, MlirIdentifier fieldName, MlirStringRef primeStr
) {
  return llzkFelt_FieldSpecAttrGet(ctx, fieldName, primeStr);
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromParts(
    MlirContext ctx, MlirIdentifier fieldName, const uint64_t *parts, intptr_t nParts
) {
  assert(nParts >= 0 && "part count must be non-negative");
  assert((parts || nParts == 0) && "non-empty parts must not be null");
  return wrap(FieldSpecAttr::get(unwrap(ctx), unwrap(fieldName), llvm::ArrayRef(parts, nParts)));
}

MlirType llzkFelt_FeltTypeGetUnspecified(MlirContext ctx) {
  return wrap(FeltType::get(unwrap(ctx)));
}

MlirType llzkFelt_FeltTypeGetFromRef(MlirContext ctx, MlirStringRef fieldName) {
  return wrap(FeltType::get(unwrap(ctx), unwrap(fieldName)));
}
