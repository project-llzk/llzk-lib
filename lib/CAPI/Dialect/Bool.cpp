//===-- Bool.cpp - Bool dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Attrs.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"

#include "llzk-c/Dialect/Bool.h"

#include <mlir/CAPI/Registration.h>

using namespace llzk::boolean;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Bool, llzk__boolean, llzk::boolean::BoolDialect)

MlirAttribute llzkFeltCmpPredicateAttrGet(MlirContext ctx, LlzkCmp cmp) {
  return wrap(FeltCmpPredicateAttr::get(unwrap(ctx), FeltCmpPredicate(cmp)));
}

bool llzkAttributeIsA_Bool_FeltCmpPredicateAttr(MlirAttribute attr) {
  return llvm::isa<FeltCmpPredicateAttr>(unwrap(attr));
}
