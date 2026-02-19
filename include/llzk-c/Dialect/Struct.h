//===-- Struct.h - C API for Struct dialect -----------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Struct dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_STRUCT_H
#define LLZK_C_DIALECT_STRUCT_H

#include "llzk-c/Support.h"

#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <stdint.h>

// Include the generated CAPI
#include "llzk/Dialect/Struct/IR/Ops.capi.h.inc"
#include "llzk/Dialect/Struct/IR/Types.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Struct, llzk__component);

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Creates a llzk::component::StructType.
/// The name attribute must be a SymbolRefAttr.
MLIR_CAPI_EXPORTED MlirType llzkStruct_StructTypeGet(MlirAttribute name);

/// Creates a llzk::component::StructType with an ArrayAttr as parameters. The name attribute must
/// be a SymbolRefAttr.
MLIR_CAPI_EXPORTED
MlirType llzkStruct_StructTypeGetWithArrayAttr(MlirAttribute name, MlirAttribute params);

/// Creates a llzk::component::StructType with an array of parameters.
/// The name attribute must be a SymbolRefAttr.
MLIR_CAPI_EXPORTED MlirType llzkStruct_StructTypeGetWithAttrs(
    MlirAttribute name, intptr_t numParams, MlirAttribute const *params
);

//===----------------------------------------------------------------------===//
// StructDefOp
//===----------------------------------------------------------------------===//

/// Returns the single body Block within the StructDefOp's Region.
MLIR_CAPI_EXPORTED MlirBlock llzkStruct_StructDefOpGetBody(MlirOperation op);

/// Returns the associated StructType to this op using the const params defined by the op.
MLIR_CAPI_EXPORTED MlirType llzkStruct_StructDefOpGetType(MlirOperation op);

/// Returns the associated StructType to this op using the given const params instead of the
/// parameters defined by the op. The const params are defined in the given attribute which has to
/// be of type ArrayAttr.
MLIR_CAPI_EXPORTED MlirType
llzkStruct_StructDefOpGetTypeWithParams(MlirOperation op, MlirAttribute params);

/// Fills the given array with the FieldDefOp operations inside this struct. The pointer to the
/// operations must have been preallocated. See `llzkStruct_StructDefOpGetNumFieldDefs` for
/// obtaining the required size of the array.
MLIR_CAPI_EXPORTED void llzkStruct_StructDefOpGetFieldDefs(MlirOperation op, MlirOperation *dst);

/// Returns the number of FieldDefOp operations defined in this struct.
MLIR_CAPI_EXPORTED intptr_t llzkStruct_StructDefOpGetNumFieldDefs(MlirOperation op);

/// Returns the header string of the struct. The size of the string is written into the given size
/// pointer. The caller is responsible of freeing the string and of providing an allocator.
MLIR_CAPI_EXPORTED const char *llzkStruct_StructDefOpGetHeaderString(
    MlirOperation op, intptr_t *dstSize, char *(*alloc_string)(size_t)
);

/// Returns true if the struct has a parameter that has the given name.
LLZK_DECLARE_NARY_OP_PREDICATE(Struct, StructDefOp, HasParamName, MlirStringRef name);

//===----------------------------------------------------------------------===//
// FieldReadOp
//===----------------------------------------------------------------------===//

/// Creates a FieldReadOp.
LLZK_DECLARE_OP_BUILD_METHOD(
    Struct, FieldReadOp, MlirType type, MlirValue component, MlirStringRef fieldName
);

/// Creates a FieldReadOp to a column offset by the given distance affine map. The values in the
/// ValueRange are operands representing the arguments to the affine map. The integer value is the
/// number of arguments in the map that are dimensions.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Struct, FieldReadOp, WithAffineMapDistance, MlirType type, MlirValue component,
    MlirStringRef fieldName, MlirAffineMap affineMap, MlirValueRange mapOperands,
    int32_t nDimensions
);

/// Creates a FieldReadOp to a column offset by the given distance defined by a name to a constant
/// parameter in the struct.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Struct, FieldReadOp, WithConstParamDistance, MlirType type, MlirValue component,
    MlirStringRef fieldName, MlirStringRef paramName
);

/// Creates a FieldReadOp to a column offset by the given distance defined by an integer value.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Struct, FieldReadOp, WithLiteralDistance, MlirType type, MlirValue component,
    MlirStringRef fieldName, int64_t distance
);

#ifdef __cplusplus
}
#endif

#endif
