//===-- Felt.cpp - Felt dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include "llzk-c/Dialect/Felt.h"

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt, FeltDialect)

MlirAttribute llzkFelt_FeltConstAttrGet(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value)));
}

MlirAttribute
llzkFelt_FeltConstAttrGetWithField(MlirContext ctx, int64_t value, MlirStringRef fieldName) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value), unwrap(fieldName)));
}

MlirAttribute llzkFelt_FeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value)));
}

MlirAttribute llzkFelt_FeltConstAttrGetWithBitsWithField(
    MlirContext ctx, unsigned numBits, int64_t value, MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value), unwrap(fieldName)));
}

MlirAttribute
llzkFelt_FeltConstAttrGetFromString(MlirContext ctx, unsigned numBits, MlirStringRef str) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromStringWithField(
    MlirContext ctx, unsigned numBits, MlirStringRef str, MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str), unwrap(fieldName)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
) {
  return wrap(FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts)));
}

MlirAttribute llzkFelt_FeltConstAttrGetFromPartsWithField(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts,
    MlirStringRef fieldName
) {
  return wrap(
      FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts), unwrap(fieldName))
  );
}

bool llzkAttributeIsA_Felt_FeltConstAttr(MlirAttribute attr) {
  return llvm::isa<FeltConstAttr>(unwrap(attr));
}

MlirType llzkFelt_FeltConstAttrGetType(MlirAttribute attr) {
  Attribute a = unwrap(attr);
  if (auto f = llvm::dyn_cast<FeltConstAttr>(a)) {
    return wrap(f.getType());
  }
  return wrap(Type {});
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromString(
    MlirContext context, MlirStringRef fieldName, unsigned numBits, MlirStringRef primeStr
) {
  auto ctx = unwrap(context);
  return wrap(
      FieldSpecAttr::get(
          ctx, StringAttr::get(ctx, unwrap(fieldName)), APInt(numBits, unwrap(primeStr), 10)
      )
  );
}

MlirAttribute llzkFelt_FieldSpecAttrGetFromParts(
    MlirContext context, MlirStringRef fieldName, unsigned numBits, const uint64_t *parts,
    intptr_t nParts
) {
  auto ctx = unwrap(context);
  return wrap(
      FieldSpecAttr::get(
          ctx, StringAttr::get(ctx, unwrap(fieldName)), APInt(numBits, ArrayRef(parts, nParts))
      )
  );
}

MlirType llzkFelt_FeltTypeGet(MlirContext ctx) { return wrap(FeltType::get(unwrap(ctx))); }

MlirType llzkFelt_FeltTypeGetWithField(MlirContext context, MlirStringRef fieldName) {
  auto ctx = unwrap(context);
  return wrap(FeltType::get(ctx, StringAttr::get(ctx, unwrap(fieldName))));
}

bool llzkTypeIsA_Felt_FeltType(MlirType type) { return llvm::isa<FeltType>(unwrap(type)); }
