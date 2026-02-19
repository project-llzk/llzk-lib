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

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Felt, llzk__felt);

//===----------------------------------------------------------------------===//
// FeltConstAttr
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FeltConstAttr with an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGet(MlirContext context, int64_t value);

/// Creates a llzk::felt::FeltConstAttr with a specified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFeltConstAttrGetWithField(MlirContext context, int64_t value, MlirStringRef fieldName);

/// Creates a llzk::felt::FeltConstAttr with a set bit length in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFeltConstAttrGetWithBits(MlirContext ctx, unsigned numBits, int64_t value);

/// Creates a llzk::felt::FeltConstAttr with a set bit length in a specified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGetWithBitsWithField(
    MlirContext ctx, unsigned numBits, int64_t value, MlirStringRef fieldName
);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a number.
/// in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFeltConstAttrGetFromString(MlirContext context, unsigned numBits, MlirStringRef str);

/// Creates a llzk::felt::FeltConstAttr from a base-10 representation of a number.
/// in a specified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGetFromStringWithField(
    MlirContext context, unsigned numBits, MlirStringRef str, MlirStringRef fieldName
);

/// Creates a llzk::felt::FeltConstAttr from an array of big-integer parts in LSB order
/// in an unspecified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGetFromParts(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts
);

/// Creates a llzk::felt::FeltConstAttr from an array of big-integer parts in LSB order
/// in a specified field.
MLIR_CAPI_EXPORTED MlirAttribute llzkFeltConstAttrGetFromPartsWithField(
    MlirContext context, unsigned numBits, const uint64_t *parts, intptr_t nParts,
    MlirStringRef fieldName
);

/// Returns true if the attribute is a FeltConstAttr.
LLZK_DECLARE_ATTR_ISA(Felt, FeltConstAttr);

/// Get the underlying felt type of the FeltConstAttr.
MLIR_CAPI_EXPORTED MlirType llzkFeltConstAttrGetType(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// FieldSpecAttr
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FieldSpecAttr from a base-10 representation of the prime.
MLIR_CAPI_EXPORTED MlirAttribute llzkFieldSpecAttrGetFromString(
    MlirContext context, MlirStringRef fieldName, unsigned numBits, MlirStringRef primeStr
);

/// Creates a llzk::felt::FieldSpecAttr from an array of big-integer parts in LSB order representing
/// the prime.
MLIR_CAPI_EXPORTED MlirAttribute llzkFieldSpecAttrGetFromParts(
    MlirContext context, MlirStringRef fieldName, unsigned numBits, const uint64_t *parts,
    intptr_t nParts
);

//===----------------------------------------------------------------------===//
// FeltType
//===----------------------------------------------------------------------===//

/// Creates a llzk::felt::FeltType with an unspecified field.
MLIR_CAPI_EXPORTED MlirType llzkFeltTypeGet(MlirContext context);

/// Creates a llzk::felt::FeltType in a given field.
MLIR_CAPI_EXPORTED MlirType llzkFeltTypeGetWithField(MlirContext context, MlirStringRef fieldName);

/// Returns true if the type is a FeltType.
LLZK_DECLARE_TYPE_ISA(Felt, FeltType);

#ifdef __cplusplus
}
#endif

#endif
