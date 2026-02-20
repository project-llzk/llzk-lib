//===-- POD.h - C API for POD dialect -----------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// POD dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_POD_H
#define LLZK_C_DIALECT_POD_H

#include "llzk-c/Support.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <stdint.h>

// Include the generated CAPI
#include "llzk/Dialect/Pod/IR/Attrs.capi.h.inc"
#include "llzk/Dialect/Pod/IR/Ops.capi.h.inc"
#include "llzk/Dialect/Pod/IR/Types.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(POD, llzk__pod);

//===----------------------------------------------------------------------===//
// Auxiliary types
//===----------------------------------------------------------------------===//

typedef struct LlzkRecordValue {
  MlirStringRef name;
  MlirValue value;
} LlzkRecordValue;

//===----------------------------------------------------------------------===//
// RecordAttr
//===----------------------------------------------------------------------===//

/// Creates a new llzk::pod::RecordAttr using the MlirContext of the given type.
MLIR_CAPI_EXPORTED MlirAttribute
llzkPod_RecordAttrGetInferredContext(MlirIdentifier name, MlirType type);

//===----------------------------------------------------------------------===//
// PodType
//===----------------------------------------------------------------------===//

/// Creates an llzk::pod::PodType using a list of values for inferring the records.
MLIR_CAPI_EXPORTED MlirType llzkPod_PodTypeGetFromInitialValues(
    MlirContext context, intptr_t nRecords, LlzkRecordValue const *records
);

/// Writes the records into the given array that must have been previously allocated
/// with enough space.
///
/// See `llzkPod_PodTypeGetRecordsCount`
MLIR_CAPI_EXPORTED void llzkPod_PodTypeGetRecords(MlirType type, MlirAttribute *dst);

/// Lookups a record type by name.
///
/// If the record is not found reports an error without any location information and
/// the returned type is NULL.
MLIR_CAPI_EXPORTED MlirType llzkPod_PodTypeLookupRecord(MlirType type, MlirStringRef name);

/// Lookups a record type by name.
///
/// If the record is not found reports an error at the given location information and
/// the returned type is NULL.
MLIR_CAPI_EXPORTED MlirType
llzkPod_PodTypeLookupRecordWithinLocation(MlirType type, MlirStringRef name, MlirLocation loc);

/// Lookups a record type by name.
///
/// If the record is not found reports an error using the given operation for location information
/// and the returned type is NULL.
MLIR_CAPI_EXPORTED MlirType
llzkPod_PodTypeLookupRecordWithinOperation(MlirType type, MlirStringRef name, MlirOperation op);

//===----------------------------------------------------------------------===//
// NewPodOp
//===----------------------------------------------------------------------===//

/// Creates a NewPodOp from a list of initialization values.
///
/// The type of the struct gets inferred from the initial values.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Pod, NewPodOp, InferredFromInitialValues, intptr_t nValues, LlzkRecordValue const *values
);

/// Creates a NewPodOp from a list of initialization values and a PodType.
///
/// The initial values can partially initialize the struct.
LLZK_DECLARE_OP_BUILD_METHOD(
    Pod, NewPodOp, MlirType type, intptr_t nValues, LlzkRecordValue const *values
);

/// Creates a NewPodOp with a list of initialization values, a PodType, and map operands.
///
/// The initial values can partially initialize the struct.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Pod, NewPodOp, WithMapOperands, MlirType type, intptr_t nValues, LlzkRecordValue const *values,
    LlzkAffineMapOperandsBuilder mapOperands
);

#ifdef __cplusplus
}
#endif

#endif
