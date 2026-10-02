//===-- Felt.h - C API for Felt dialect ---------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Felt dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_FELT_H
#define LLZK_C_DIALECT_FELT_H

#include <mlir-c/IR.h>

// Include the generated CAPI
#include "llzk/Dialect/Felt/IR/Attrs.capi.h.inc"
#include "llzk/Dialect/Felt/IR/Ops.capi.h.inc"
#include "llzk/Dialect/Felt/IR/Types.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

/// Get reference to the LLZK `felt` dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt);

//===----------------------------------------------------------------------===//
// FeltConstAttr
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FeltConstAttr with the given value in the specified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetInField(MlirContext ctx, int64_t value, MlirStringRef fieldName);

/// Creates a llzk::felt::FeltConstAttr with the given value in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetUnspecified(MlirContext ctx, int64_t value);

//===----------------------------------------------------------------------===//
// FeltType
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FeltType with an unspecified field.
MLIR_CAPI_EXPORTED MlirType llzkFelt_FeltTypeGetUnspecified(MlirContext ctx);

/// Create a llzk::felt::FeltType Type with the given parameters.
MLIR_CAPI_EXPORTED MlirType llzkFelt_FeltTypeGetFromRef(MlirContext ctx, MlirStringRef fieldName);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECT_FELT_H
