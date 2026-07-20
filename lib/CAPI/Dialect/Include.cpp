//===-- Include.cpp - Include dialect C API implementation ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Include.h"

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"

#include <mlir-c/Pass.h>

#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace llzk::include;

static inline void registerLLZKIncludeTransformationPasses() { registerTransformationPasses(); }

// Include the generated CAPI
#include "llzk/Dialect/Include/IR/Ops.capi.cpp.inc"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Include, llzk__include, IncludeDialect)

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Include, IncludeOp, InferredContext, MlirStringRef name, MlirStringRef path
) {
  return mlirOpBuilderInsert(
      builder, wrap(llzk::create<IncludeOp>(builder, location, unwrap(name), unwrap(path)))
  );
}
