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

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Struct, llzk__component);

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

/// Creates a llzk::component::StructType.
/// The name attribute must be a SymbolRefAttr.
MLIR_CAPI_EXPORTED MlirType llzkStructTypeGet(MlirAttribute name);

/// Creates a llzk::component::StructType with an ArrayAttr as parameters. The name attribute must
/// be a SymbolRefAttr.
MLIR_CAPI_EXPORTED
MlirType llzkStructTypeGetWithArrayAttr(MlirAttribute name, MlirAttribute params);

/// Creates a llzk::component::StructType with an array of parameters.
/// The name attribute must be a SymbolRefAttr.
MLIR_CAPI_EXPORTED MlirType
llzkStructTypeGetWithAttrs(MlirAttribute name, intptr_t numParams, MlirAttribute const *params);

/// Returns true if the type is a StructType.
LLZK_DECLARE_TYPE_ISA(Struct, StructType);

/// Returns the fully qualified name of a llzk::component::StructType.
MLIR_CAPI_EXPORTED MlirAttribute llzkStructTypeGetName(MlirType type);

/// Returns the parameter of a llzk::component::StructType as an ArrayAttr.
MLIR_CAPI_EXPORTED MlirAttribute llzkStructTypeGetParams(MlirType type);

/// Lookups the definition Operation of the given StructType using the given
/// Operation as root for the lookup. The definition Operation is wrapped
/// in a LlzkSymbolLookupResult that the caller is responsible for cleaning up.
///
/// If the function returns 'success' the lookup result will be stored in the
/// given pointer. Accessing the lookup result if the function returns 'failure'
/// is undefined behavior.
///
/// Requires that the given Operation implements the SymbolTable op interface.
MLIR_CAPI_EXPORTED MlirLogicalResult llzkStructStructTypeGetDefinition(
    MlirType type, MlirOperation root, LlzkSymbolLookupResult *result
);

/// Lookups the definition Operation of the given StructType using the given
/// Module as root for the lookup. The definition Operation is wrapped
/// in a LlzkSymbolLookupResult that the caller is responsible for cleaning up.
///
/// If the function returns 'success' the lookup result will be stored in the
/// given pointer. Accessing the lookup result if the function returns 'failure'
/// is undefined behavior.
MLIR_CAPI_EXPORTED MlirLogicalResult llzkStructStructTypeGetDefinitionFromModule(
    MlirType type, MlirModule root, LlzkSymbolLookupResult *result
);

//===----------------------------------------------------------------------===//
// StructDefOp
//===----------------------------------------------------------------------===//

/// Returns true if the op is a StructDefOp
LLZK_DECLARE_OP_ISA(Struct, StructDefOp);

/// Returns the single body Region of the StructDefOp.
MLIR_CAPI_EXPORTED MlirRegion llzkStructDefOpGetBodyRegion(MlirOperation op);

/// Returns the single body Block within the StructDefOp's Region.
MLIR_CAPI_EXPORTED MlirBlock llzkStructDefOpGetBody(MlirOperation op);

/// Returns the associated StructType to this op using the const params defined by the op.
MLIR_CAPI_EXPORTED MlirType llzkStructDefOpGetType(MlirOperation op);

/// Returns the associated StructType to this op using the given const params instead of the
/// parameters defined by the op. The const params are defined in the given attribute which has to
/// be of type ArrayAttr.
MLIR_CAPI_EXPORTED MlirType
llzkStructDefOpGetTypeWithParams(MlirOperation op, MlirAttribute params);

/// Returns the operation that defines the member with the given name, if present.
MLIR_CAPI_EXPORTED MlirOperation llzkStructDefOpGetMemberDef(MlirOperation op, MlirStringRef name);

/// Fills the given array with the MemberDefOp operations inside this struct. The pointer to the
/// operations must have been preallocated. See `llzkStructDefOpGetNumMemberDefs` for obtaining the
/// required size of the array.
MLIR_CAPI_EXPORTED void llzkStructDefOpGetMemberDefs(MlirOperation op, MlirOperation *dst);

/// Returns the number of MemberDefOp operations defined in this struct.
MLIR_CAPI_EXPORTED intptr_t llzkStructDefOpGetNumMemberDefs(MlirOperation op);

/// Returns true if the struct has members marked as columns.
MlirLogicalResult llzkStructDefOpGetHasColumns(MlirOperation op);

/// Returns the FuncDefOp operation that defines the witness computation of the struct.
MLIR_CAPI_EXPORTED MlirOperation llzkStructDefOpGetComputeFuncOp(MlirOperation op);

/// Returns the FuncDefOp operation that defines the constraints of the struct.
MLIR_CAPI_EXPORTED MlirOperation llzkStructDefOpGetConstrainFuncOp(MlirOperation op);

/// Returns the header string of the struct. The size of the string is written into the given size
/// pointer. The caller is responsible of freeing the string and of providing an allocator.
MLIR_CAPI_EXPORTED const char *
llzkStructDefOpGetHeaderString(MlirOperation op, intptr_t *dstSize, char *(*alloc_string)(size_t));

/// Returns true if the struct has a parameter that with the given name.
LLZK_DECLARE_NARY_OP_PREDICATE(StructDefOp, HasParamName, MlirStringRef name);

/// Returns a StringAttr with the fully qualified name of the struct.
MLIR_CAPI_EXPORTED MlirAttribute llzkStructDefOpGetFullyQualifiedName(MlirOperation op);

/// Returns true if the struct is the main entry point of the circuit.
LLZK_DECLARE_OP_PREDICATE(StructDefOp, IsMainComponent);

//===----------------------------------------------------------------------===//
// MemberDefOp
//===----------------------------------------------------------------------===//

/// Returns true if the op is a MemberDefOp
LLZK_DECLARE_OP_ISA(Struct, MemberDefOp);

/// Returns true if the member has been marked public with a PublicAttr
LLZK_DECLARE_OP_PREDICATE(MemberDefOp, HasPublicAttr);

/// Sets the public attribute in the given member.
MLIR_CAPI_EXPORTED void llzkMemberDefOpSetPublicAttr(MlirOperation op, bool value);

//===----------------------------------------------------------------------===//
// MemberReadOp
//===----------------------------------------------------------------------===//

/// Creates a MemberReadOp.
LLZK_DECLARE_OP_BUILD_METHOD(
    Struct, MemberReadOp, MlirType type, MlirValue component, MlirStringRef memberName
);

/// Creates a MemberReadOp to a column offset by the given distance affine map. The values in the
/// ValueRange are operands representing the arguments to the affine map. The integer value is the
/// number of arguments in the map that are dimensions.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Struct, MemberReadOp, WithAffineMapDistance, MlirType type, MlirValue component,
    MlirStringRef memberName, MlirAffineMap affineMap, MlirValueRange mapOperands
);

/// Creates a MemberReadOp to a column offset by the given distance defined by a name to a constant
/// parameter in the struct.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Struct, MemberReadOp, WithConstParamDistance, MlirType type, MlirValue component,
    MlirStringRef memberName, MlirStringRef paramName
);

/// Creates a MemberReadOp to a column offset by the given distance defined by an integer value.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Struct, MemberReadOp, WithLiteralDistance, MlirType type, MlirValue component,
    MlirStringRef memberName, int64_t distance
);

#ifdef __cplusplus
}
#endif

#endif
