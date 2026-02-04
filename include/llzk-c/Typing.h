//===-- Typing.h - C API for llzk types ---------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares functions that check for properties of types
// in different llzk constructs.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_TYPING_H
#define LLZK_C_TYPING_H

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

/// This function asserts that the given Attribute kind is legal within the LLZK types that can
/// contain Attribute parameters (i.e., ArrayType, StructType, and TypeVarType). This should be used
/// in any function that examines the attribute parameters within parameterized LLZK types to ensure
/// that the function handles all possible cases properly, especially if more legal attributes are
/// added in the future. Throw a fatal error if anything illegal is found, indicating that the
/// caller of this function should be updated.
MLIR_CAPI_EXPORTED void llzkAssertValidAttrForParamOfType(MlirAttribute attr);

/// valid types: {I1, Index, String, FeltType, StructType, ArrayType, TypeVarType}
MLIR_CAPI_EXPORTED bool llzkIsValidType(MlirType type);

/// valid types: {FeltType, StructType (with columns), ArrayType (that contains a valid column
/// type)}
MLIR_CAPI_EXPORTED bool llzkIsValidColumnType(MlirType type, MlirOperation op);

/// valid types: isValidType() - {TypeVarType} - {types with variable parameters}
MLIR_CAPI_EXPORTED bool llzkIsValidGlobalType(MlirType type);

/// valid types: isValidType() - {String, StructType} (excluded via any type parameter nesting)
MLIR_CAPI_EXPORTED bool llzkIsValidEmitEqType(MlirType type);

/// valid types: {I1, Index, FeltType, TypeVarType}
MLIR_CAPI_EXPORTED bool llzkIsValidConstReadType(MlirType type);

/// valid types: isValidType() - {ArrayType}
MLIR_CAPI_EXPORTED bool llzkIsValidArrayElemType(MlirType type);

/// Checks if the type is a LLZK Array and it also contains a valid LLZK type.
MLIR_CAPI_EXPORTED bool llzkIsValidArrayType(MlirType type);

/// Return `false` if the type contains any of the following:
/// - `TypeVarType`
/// - `SymbolRefAttr`
/// - `AffineMapAttr`
/// - `StructType` with parameters if `allowStructParams==false`
MLIR_CAPI_EXPORTED bool llzkIsConcreteType(MlirType type, bool allowStructParams);

/// @brief Return `true` iff the given type contains an AffineMapAttr.
MLIR_CAPI_EXPORTED bool llzkHasAffineMapAttr(MlirType type);

/// Return `true` iff the two ArrayRef instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
MLIR_CAPI_EXPORTED bool llzkTypeParamsUnify(
    intptr_t, MlirAttribute const *lhsParams, intptr_t, MlirAttribute const *rhsParams
);

/// Return `true` iff the two ArrayAttr instances containing StructType or ArrayType parameters
/// are equivalent or could be equivalent after full instantiation of struct parameters.
MLIR_CAPI_EXPORTED bool
llzkArrayAttrTypeParamsUnify(MlirAttribute lhsParams, MlirAttribute rhsParams);

/// Return `true` iff the two Type instances are equivalent or could be equivalent after full
/// instantiation of struct parameters (if applicable within the given types).
MLIR_CAPI_EXPORTED bool llzkTypesUnify(
    MlirType lhs, MlirType rhs, intptr_t nRhsReversePrefix, MlirStringRef const *rhsReversePrefix
);

/// Return `true` iff the types unify and `newTy` is "more concrete" than `oldTy`.
///
/// The types `i1`, `index`, `felt.type`, and `string.type` are concrete whereas `poly.tvar` is
/// not (because it may be substituted with any type during struct instantiation). When considering
/// the attributes with `array.type` and `struct.type` types, we define IntegerAttr and TypeAttr
/// as concrete, AffineMapAttr as less concrete than those, and SymbolRefAttr as least concrete.
MLIR_CAPI_EXPORTED bool llzkIsMoreConcreteUnification(
    MlirType oldTy, MlirType newTy, bool (*knownOldToNew)(MlirType, MlirType, void *), void *data
);

/// Convert any IntegerAttr with a type other than IndexType to use IndexType.
/// The location is used for error reporting if the conversion is not possible.
MLIR_CAPI_EXPORTED MlirAttribute llzkForceIntAttrType(MlirAttribute attr, MlirLocation loc);

#ifdef __cplusplus
}
#endif

#endif
