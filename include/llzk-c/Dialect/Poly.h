//===-- Poly.h - C API for Polymorphic dialect --------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Polymorphic dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_POLYMORPHIC_H
#define LLZK_C_DIALECT_POLYMORPHIC_H

#include "llzk-c/Support.h"

#include <mlir-c/AffineExpr.h>
#include <mlir-c/AffineMap.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

// Include the generated CAPI
#include "llzk/Dialect/Polymorphic/IR/Ops.capi.h.inc"
#include "llzk/Dialect/Polymorphic/IR/Types.capi.h.inc"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Polymorphic, llzk__polymorphic);

//===----------------------------------------------------------------------===//
// TypeVarType
//===----------------------------------------------------------------------===//

/// Creates a llzk::polymorphic::TypeVarType.
MLIR_CAPI_EXPORTED MlirType
llzkPoly_TypeVarTypeGetFromStringRef(MlirContext context, MlirStringRef value);

/// Creates a llzk::polymorphic::TypeVarType from either a StringAttr or a FlatSymbolRefAttr.
MLIR_CAPI_EXPORTED MlirType llzkPoly_TypeVarTypeGetFromAttr(MlirAttribute value);

//===----------------------------------------------------------------------===//
// ApplyMapOp
//===----------------------------------------------------------------------===//

/// Creates an ApplyMapOp with the given attribute that has to be of type AffineMapAttr.
LLZK_DECLARE_OP_BUILD_METHOD(
    Poly, ApplyMapOp, MlirAttribute affineMapAttr, MlirValueRange operands
);

/// Creates an ApplyMapOp with the given affine map.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Poly, ApplyMapOp, WithAffineMap, MlirAffineMap affineMap, MlirValueRange operands
);

/// Creates an ApplyMapOp with the given affine expression.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Poly, ApplyMapOp, WithAffineExpr, MlirAffineExpr affineExpr, MlirValueRange operands
);

/// Returns the number of operands that correspond to dimensions in the affine map.
MLIR_CAPI_EXPORTED intptr_t llzkPoly_ApplyMapOpGetNumDimOperands(MlirOperation op);

/// Writes into the destination buffer the operands that correspond to dimensions in the affine map.
/// The buffer needs to be preallocated first with the necessary amount and the caller is
/// responsible of its lifetime. See `llzkPoly_ApplyMapOpGetNumDimOperands`.
MLIR_CAPI_EXPORTED void llzkPoly_ApplyMapOpGetDimOperands(MlirOperation op, MlirValue *dst);

/// Returns the number of operands that correspond to symbols in the affine map.
MLIR_CAPI_EXPORTED intptr_t llzkPoly_ApplyMapOpGetNumSymbolOperands(MlirOperation op);

/// Writes into the destination buffer the operands that correspond to symbols in the affine map.
/// The buffer needs to be preallocated first with the necessary amount and the caller is
/// responsible of its lifetime. See `llzkPoly_ApplyMapOpGetNumSymbolOperands`.
MLIR_CAPI_EXPORTED void llzkPoly_ApplyMapOpGetSymbolOperands(MlirOperation op, MlirValue *dst);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECT_POLYMORPHIC_H
