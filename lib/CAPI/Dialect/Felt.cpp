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

MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetInField(MlirContext ctx, int64_t value, MlirStringRef fieldName) {
  return wrap(FeltConstAttr::get(unwrap(ctx), llvm::DynamicAPInt(value), unwrap(fieldName)));
}

MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetUnspecified(MlirContext ctx, int64_t value) {
  return wrap(FeltConstAttr::get(unwrap(ctx), llvm::DynamicAPInt(value)));
}

MlirType llzkFelt_FeltTypeGetUnspecified(MlirContext ctx) {
  return wrap(FeltType::get(unwrap(ctx)));
}

MlirType llzkFelt_FeltTypeGetFromRef(MlirContext ctx, MlirStringRef fieldName) {
  return wrap(FeltType::get(unwrap(ctx), unwrap(fieldName)));
}
