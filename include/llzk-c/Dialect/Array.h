//===-- Array.h - C API for Array dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Array dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_ARRAY_H
#define LLZK_C_DIALECT_ARRAY_H

#include "llzk/Dialect/Array/Transforms/TransformationPasses.capi.h.inc"

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Array, llzk__array);

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Creates an llzk::array::ArrayType using a list of attributes as dimensions.
MLIR_CAPI_EXPORTED MlirType
llzkArrayTypeGet(MlirType type, intptr_t nDims, MlirAttribute const *dims);

/// Returns true if the type is an llzk::array::ArrayType.
LLZK_DECLARE_TYPE_ISA(Array, ArrayType);

/// Creates an llzk::array::ArrayType using a list of numbers as dimensions.
MLIR_CAPI_EXPORTED MlirType
llzkArrayTypeGetWithNumericDims(MlirType type, intptr_t nDims, int64_t const *dims);

/// Returns the element type of an llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirType llzkArrayTypeGetElementType(MlirType type);

/// Returns the number of dimensions of an llzk::array::ArrayType.
MLIR_CAPI_EXPORTED intptr_t llzkArrayTypeGetNumDims(MlirType type);

/// Returns the n-th dimention of an llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirAttribute llzkArrayTypeGetDim(MlirType type, intptr_t dim);

//===----------------------------------------------------------------------===//
// CreateArrayOp
//===----------------------------------------------------------------------===//

/// Creates a CreateArrayOp from a list of Values.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Array, CreateArrayOp, WithValues, MlirType arrType, intptr_t nValues, MlirValue const *values
);

/// Creates a CreateArrayOp with its size information declared with AffineMaps and operands.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Array, CreateArrayOp, WithMapOperands, MlirType arrType,
    LlzkAffineMapOperandsBuilder mapOperands
);

#ifdef __cplusplus
}
#endif

#endif
