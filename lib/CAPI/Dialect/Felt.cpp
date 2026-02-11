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

MlirAttribute llzkFeltConstAttrGet(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value)));
}

MlirAttribute
llzkFeltConstAttrGetWithField(MlirContext ctx, int64_t value, MlirStringRef fieldName) {
  return wrap(FeltConstAttr::get(unwrap(ctx), toAPInt(value), unwrap(fieldName)));
}

MlirAttribute llzkFeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value)));
}

MlirAttribute llzkFeltConstAttrGetWithBitsWithField(
    MlirContext ctx, unsigned numBits, int64_t value, MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), APInt(numBits, value), unwrap(fieldName)));
}

MlirAttribute llzkFeltConstAttrGetFromString(MlirContext ctx, unsigned numBits, MlirStringRef str) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str)));
}

MlirAttribute llzkFeltConstAttrGetFromStringWithField(
    MlirContext ctx, unsigned numBits, MlirStringRef str, MlirStringRef fieldName
) {
  return wrap(FeltConstAttr::get(unwrap(ctx), numBits, unwrap(str), unwrap(fieldName)));
}

MlirAttribute llzkFeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
) {
  return wrap(FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts)));
}

MlirAttribute llzkFeltConstAttrGetFromPartsWithField(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts,
    MlirStringRef fieldName
) {
  return wrap(
      FeltConstAttr::get(unwrap(context), numBits, ArrayRef(parts, nParts), unwrap(fieldName))
  );
}

bool llzkAttributeIsAFeltConstAttr(MlirAttribute attr) {
  return llvm::isa<FeltConstAttr>(unwrap(attr));
}

MlirType llzkFeltConstAttrGetType(MlirAttribute attr) {
  Attribute a = unwrap(attr);
  if (auto f = llvm::dyn_cast<FeltConstAttr>(a)) {
    return wrap(f.getType());
  }
  return wrap(Type {});
}

MlirAttribute llzkFieldSpecAttrGetFromString(
    MlirContext context, MlirStringRef fieldName, unsigned numBits, MlirStringRef primeStr
) {
  auto ctx = unwrap(context);
  return wrap(
      FieldSpecAttr::get(
          ctx, StringAttr::get(ctx, unwrap(fieldName)), APInt(numBits, unwrap(primeStr), 10)
      )
  );
}

MlirAttribute llzkFieldSpecAttrGetFromParts(
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

MlirType llzkFeltTypeGet(MlirContext ctx) { return wrap(FeltType::get(unwrap(ctx))); }

MlirType llzkFeltTypeGetWithField(MlirContext context, MlirStringRef fieldName) {
  auto ctx = unwrap(context);
  return wrap(FeltType::get(ctx, StringAttr::get(ctx, unwrap(fieldName))));
}

bool llzkTypeIsAFeltType(MlirType type) { return llvm::isa<FeltType>(unwrap(type)); }
