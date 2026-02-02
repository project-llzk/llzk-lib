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

/// Creates a new llzk::pod::RecordAttr.
MLIR_CAPI_EXPORTED MlirAttribute llzkRecordAttrGet(MlirStringRef name, MlirType type);

/// Returns true if the attribute is an llzk::pod::RecordAttr.
LLZK_DECLARE_ATTR_ISA(RecordAttr);

/// Returns the name of the record.
MLIR_CAPI_EXPORTED MlirStringRef llzkRecordAttrGetName(MlirAttribute attr);

/// Returns the name of the record as a flat symbol attribute.
MLIR_CAPI_EXPORTED MlirAttribute llzkRecordAttrGetNameSym(MlirAttribute attr);

/// Returns the type of the record.
MLIR_CAPI_EXPORTED MlirType llzkRecordAttrGetType(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// PodType
//===----------------------------------------------------------------------===//

/// Creates an llzk::pod::PodType using a list of attributes as records.
MLIR_CAPI_EXPORTED MlirType
llzkPodTypeGet(MlirContext context, intptr_t nRecords, MlirAttribute const *records);

/// Creates an llzk::pod::PodType using a list of values for inferring the records.
MLIR_CAPI_EXPORTED MlirType llzkPodTypeGetFromInitialValues(
    MlirContext context, intptr_t nRecords, LlzkRecordValue const *records
);

/// Returns true if the type is an llzk::pod::PodType.
LLZK_DECLARE_TYPE_ISA(PodType);

/// Returns the number of records in the struct.
MLIR_CAPI_EXPORTED intptr_t llzkPodTypeGetNumRecords(MlirType type);

/// Writes the records into the given array that must have been previously allocated with enough
/// space.
///
/// See `llzkPodTypeGetNumRecords`
MLIR_CAPI_EXPORTED void llzkPodTypeGetRecords(MlirType type, MlirAttribute *dst);

/// Returns the n-th record in the struct.
MLIR_CAPI_EXPORTED MlirAttribute llzkPodTypeGetNthRecord(MlirType type, intptr_t n);

/// Lookups a record type by name.
///
/// If the record is not found reports an error without any location information and
/// the returned type is NULL.
MLIR_CAPI_EXPORTED MlirType llzkPodTypeLookupRecord(MlirType type, MlirStringRef name);

/// Lookups a record type by name.
///
/// If the record is not found reports an error at the given location information and
/// the returned type is NULL.
MLIR_CAPI_EXPORTED MlirType
llzkPodTypeLookupRecordWithinLocation(MlirType type, MlirStringRef name, MlirLocation loc);

/// Lookups a record type by name.
///
/// If the record is not found reports an error using the given operation for location information
/// and the returned type is NULL.
MLIR_CAPI_EXPORTED MlirType
llzkPodTypeLookupRecordWithinOperation(MlirType type, MlirStringRef name, MlirOperation op);

//===----------------------------------------------------------------------===//
// NewPodOp
//===----------------------------------------------------------------------===//

/// Creates a NewPodOp from a list of initialization values.
///
/// The type of the struct gets inferred from the initial values.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    NewPodOp, InferredFromInitialValues, intptr_t nValues, LlzkRecordValue const *values
);

/// Creates a NewPodOp from a list of initialization values and a PodType.
///
/// The initial values can partially initialize the struct.
LLZK_DECLARE_OP_BUILD_METHOD(
    NewPodOp, MlirType type, intptr_t nValues, LlzkRecordValue const *values
);

/// Creates a NewPodOp with a list of initialization values, a PodType, and map operands.
///
/// The initial values can partially initialize the struct.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    NewPodOp, WithMapOperands, MlirType type, intptr_t nValues, LlzkRecordValue const *values,
    LlzkAffineMapOperandsBuilder mapOperands
);

/// Returns true if the op is a llzk::pod::NewPodOp.
LLZK_DECLARE_OP_ISA(NewPodOp);

//===----------------------------------------------------------------------===//
// ReadPodOp
//===----------------------------------------------------------------------===//

/// Returns true if the op is a llzk::pod::ReadPodOp.
LLZK_DECLARE_OP_ISA(ReadPodOp);

//===----------------------------------------------------------------------===//
// WritePodOp
//===----------------------------------------------------------------------===//

/// Returns true if the op is a llzk::pod::WritePodOp.
LLZK_DECLARE_OP_ISA(WritePodOp);

#ifdef __cplusplus
}
#endif

#endif
