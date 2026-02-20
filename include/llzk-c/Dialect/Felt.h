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

/// Creates a llzk::felt::FeltConstAttr with an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetUnspecified(MlirContext context, int64_t value);

/// Creates a llzk::felt::FeltConstAttr with a set bit length in a specified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetWithBits(
    MlirContext ctx, unsigned numBits, int64_t value, MlirIdentifier fieldName
);

/// Creates a llzk::felt::FeltConstAttr with a set bit length in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetWithBitsUnspecified(MlirContext ctx, unsigned numBits, int64_t value);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a number.
/// in a specified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromString(
    MlirContext context, unsigned numBits, MlirStringRef str, MlirIdentifier fieldName
);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a number.
/// in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromStringUnspecified(
    MlirContext context, unsigned numBits, MlirStringRef str
);

/// Creates a llzk::felt::FeltConstAttr from an array of big-integer parts in LSB order
/// in a specified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts,
    MlirIdentifier fieldName
);

/// Creates a llzk::felt::FeltConstAttr from an array of big-integer parts in LSB order
/// in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromPartsUnspecified(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
);

//===----------------------------------------------------------------------===//
// FieldSpecAttr
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FieldSpecAttr from a base-10 representation of the prime.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FieldSpecAttrGetFromString(
    MlirContext context, MlirIdentifier fieldName, unsigned numBits, MlirStringRef primeStr
);

/// Creates a llzk::felt::FieldSpecAttr from an array of big-integer parts in LSB order representing
/// the prime.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FieldSpecAttrGetFromParts(
    MlirContext context, MlirIdentifier fieldName, unsigned numBits, const uint64_t *parts,
    intptr_t nParts
);

//===----------------------------------------------------------------------===//
// FeltType
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FeltType with an unspecified field.
MLIR_CAPI_EXPORTED MlirType llzkFelt_FeltTypeGetUnspecified(MlirContext context);

#ifdef __cplusplus
}
#endif

#endif
