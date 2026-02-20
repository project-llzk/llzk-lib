//===-- Cast.h - C API for Cast dialect ---------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Cast dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_CAST_H
#define LLZK_C_DIALECT_CAST_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

// Include the generated CAPI
#include "llzk/Dialect/Cast/IR/Ops.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Cast, llzk__cast);

/// Creates a IntToFeltOp from an input value with the specified result FeltType.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Cast, IntToFeltOp, WithType, MlirType feltType, MlirValue value
);

#ifdef __cplusplus
}
#endif

#endif
