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

/// Creates a llzk::felt::FeltConstAttr from a signed 64-bit integer in the specified field.
/// The provided `MlirType` must be a llzk::felt::FeltType.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetFromInt64(MlirContext ctx, int64_t value, MlirType type);

/// Creates a llzk::felt::FeltConstAttr from a signed 64-bit integer in the specified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetFromInt64InField(MlirContext ctx, int64_t value, MlirStringRef fieldName);

/// Creates a llzk::felt::FeltConstAttr from a signed 64-bit integer in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetFromInt64Unspecified(MlirContext ctx, int64_t value);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a signed integer
/// in the specified field. Returns a null attribute for malformed decimal input.
/// The provided `MlirType` must be a llzk::felt::FeltType.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetFromString(MlirContext ctx, MlirStringRef str, MlirType type);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a signed integer
/// in the specified field. Returns a null attribute for malformed decimal input.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromStringInField(
    MlirContext ctx, MlirStringRef str, MlirStringRef fieldName
);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a signed integer
/// in an unspecified field. Returns a null attribute for malformed decimal input.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFelt_FeltConstAttrGetFromStringUnspecified(MlirContext ctx, MlirStringRef str);

/// Creates a llzk::felt::FeltConstAttr from unsigned 64-bit parts in LSB order using all bits. An
/// empty array represents zero. The constant is created in the specified field.
///
/// Requirements:
/// `nParts` must be non-negative and `parts` must be non-null when `nParts` is positive.
/// The provided `MlirType` must be a llzk::felt::FeltType.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromParts(
    MlirContext ctx, const uint64_t *parts, intptr_t nParts, MlirType type
);

/// Creates a llzk::felt::FeltConstAttr from unsigned 64-bit parts in LSB order using all bits. An
/// empty array represents zero. The constant is created in the specified field.
///
/// Requirements:
/// `nParts` must be non-negative and `parts` must be non-null when `nParts` is positive.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromPartsInField(
    MlirContext ctx, const uint64_t *parts, intptr_t nParts, MlirStringRef fieldName
);

/// Creates a llzk::felt::FeltConstAttr from unsigned 64-bit parts in LSB order using all bits. An
/// empty array represents zero. The constant is created in an unspecified field.
///
/// Requirements:
/// `nParts` must be non-negative and `parts` must be non-null when `nParts` is positive.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FeltConstAttrGetFromPartsUnspecified(
    MlirContext ctx, const uint64_t *parts, intptr_t nParts
);

//===----------------------------------------------------------------------===//
// FieldSpecAttr
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FieldSpecAttr from a base-10 representation of the prime.
/// Returns a null attribute for malformed decimal input.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FieldSpecAttrGetFromString(
    MlirContext ctx, MlirIdentifier fieldName, MlirStringRef primeStr
);

/// Creates a llzk::felt::FieldSpecAttr from an array of big-integer parts in LSB order representing
/// the prime. All bits are interpreted unsigned. An empty array represents zero.
///
/// Requirements:
/// `nParts` must be non-negative and `parts` must be non-null when `nParts` is positive.
MLIR_CAPI_EXPORTED MlirAttribute llzkFelt_FieldSpecAttrGetFromParts(
    MlirContext ctx, MlirIdentifier fieldName, const uint64_t *parts, intptr_t nParts
);

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
