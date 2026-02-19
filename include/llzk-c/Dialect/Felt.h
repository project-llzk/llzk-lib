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

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt);

//===----------------------------------------------------------------------===//
// FeltConstAttr
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FeltConstAttr with a set bit length.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a number.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetFromString(MlirContext context, unsigned numBits, MlirStringRef str);

/// Creates a llzk::felt::FeltConstAttr from an array of big-integer parts in LSB order.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
);

#ifdef __cplusplus
}
#endif

#endif
