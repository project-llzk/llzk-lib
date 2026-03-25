//===-- Cast.cpp - Cast dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Cast.h"
#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Ops.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>

using namespace llzk;
using namespace llzk::cast;

// Include the generated CAPI
#include "llzk/Dialect/Cast/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Cast, llzk__cast, CastDialect)

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Cast, IntToFeltOp, WithType, MlirType feltType, MlirValue value
) {
  return wrap(create<IntToFeltOp>(builder, location, unwrap(feltType), unwrap(value)));
}
