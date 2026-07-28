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

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#include <stdint.h>

// Include the generated CAPI
#include "llzk/Dialect/Array/IR/Ops.capi.h.inc"
#include "llzk/Dialect/Array/IR/Types.capi.h.inc"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

/// Get reference to the LLZK `array` dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Array, llzk__array);

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

/// Creates an llzk::array::ArrayType using a list of attributes as dimensions.
///
/// It can only be instantiated with the following element types:
///   - Any LLZK type other than an ArrayType (i.e., arrays cannot be nested)
///   - IndexType
///   - Unsigned integers of 1 bit (aka booleans)
///
/// The dimensions of the array are specified using a list of attributes, one
/// per dimension. Each attribute must be one of the following:
///   - IntegerAttr (with IndexType), specifying a fixed dimension size
///   - SymbolRefAttr, specifying a dimension size defined by a struct parameter or
///     global constant
///   - AffineMapAttr, specifying a dimension size computed from surrounding loop
///     induction variables
MLIR_CAPI_EXPORTED MlirType
llzkArray_ArrayTypeGetWithDims(MlirType type, intptr_t nDims, MlirAttribute const *dims);

/// Creates an llzk::array::ArrayType using a list of numbers as dimensions.
MLIR_CAPI_EXPORTED MlirType
llzkArray_ArrayTypeGetWithShape(MlirType type, intptr_t nDims, int64_t const *dims);

/// Returns the element type of an llzk::array::ArrayType.
MLIR_CAPI_EXPORTED MlirType llzkArray_ArrayTypeGetElementType(MlirType type);

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

#endif // LLZK_C_DIALECT_ARRAY_H
