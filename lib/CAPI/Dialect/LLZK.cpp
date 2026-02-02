//===-- LLZK.cpp - LLZK dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include "llzk-c/Dialect/LLZK.h"

#include <mlir/CAPI/Registration.h>

using namespace llzk;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLZK, llzk, LLZKDialect)

MlirAttribute llzkPublicAttrGet(MlirContext ctx) { return wrap(PublicAttr::get(unwrap(ctx))); }

bool llzkAttributeIsAPublicAttr(MlirAttribute attr) { return llvm::isa<PublicAttr>(unwrap(attr)); }

MlirAttribute llzkLoopBoundsAttrGet(MlirContext ctx, int64_t lower, int64_t upper, int64_t step) {
  return wrap(LoopBoundsAttr::get(unwrap(ctx), toAPInt(lower), toAPInt(upper), toAPInt(step)));
}

bool llzkAttributeIsALoopBoundsAttr(MlirAttribute attr) {
  return llvm::isa<LoopBoundsAttr>(unwrap(attr));
}

bool llzkOperationIsANonDetOp(MlirOperation op) { return llvm::isa<NonDetOp>(unwrap(op)); }
