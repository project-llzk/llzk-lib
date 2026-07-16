//===-- Function.h - C API for Function dialect -------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Function dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_FUNCTION_H
#define LLZK_C_DIALECT_FUNCTION_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <stdint.h>

// Include the generated CAPI
#include "llzk/Dialect/Function/IR/Attrs.capi.h.inc"
#include "llzk/Dialect/Function/IR/Ops.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

/// Get reference to the LLZK `function` dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Function, llzk__function);

//===----------------------------------------------------------------------===//
// FuncDefOp
//===----------------------------------------------------------------------===//

/// Builds a FuncDefOp with the given attributes and argument attributes. Each argument attribute
/// must be a DictionaryAttr.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithAttrsAndArgAttrs, MlirStringRef name, MlirType type, intptr_t nAttrs,
    MlirNamedAttribute const *attrs, intptr_t nArgAttrs, MlirAttribute const *argAttrs
);

/// Builds a FuncDefOp with the given attributes.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithAttrs, MlirStringRef name, MlirType type, intptr_t nAttrs,
    MlirNamedAttribute const *attrs
);

/// Builds a FuncDefOp with the given argument attributes. Each argument attribute must be a
/// DictionaryAttr.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithArgAttrs, MlirStringRef name, MlirType type, intptr_t nArgAttrs,
    MlirAttribute const *argAttrs
);

/// Builds a FuncDefOp.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithoutAttrs, MlirStringRef name, MlirType type
);

/// Returns the `function.arg_name` StringAttr for the argument at the given index, or null if the
/// argument has no `function.arg_name` attribute.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFunction_FuncDefOpGetArgNameAttr(MlirOperation op, unsigned index);

/// Sets the `function.arg_name` attribute for the argument at the given index.
///
/// The attribute must be a StringAttr. Empty and duplicate names are rejected by the FuncDefOp
/// verifier.
MLIR_CAPI_EXPORTED void
llzkFunction_FuncDefOpSetArgNameAttr(MlirOperation op, unsigned index, MlirAttribute attr);

/// Sets the `function.arg_name` attribute for the argument at the given index from a string value.
///
/// Empty and duplicate names are rejected by the FuncDefOp verifier.
MLIR_CAPI_EXPORTED void
llzkFunction_FuncDefOpSetArgName(MlirOperation op, unsigned index, MlirStringRef name);

/// Returns the `function.res_name` StringAttr for the result at the given index, or null if the
/// result has no `function.res_name` attribute.
MLIR_CAPI_EXPORTED MlirAttribute
llzkFunction_FuncDefOpGetResNameAttr(MlirOperation op, unsigned index);

/// Sets the `function.res_name` attribute for the result at the given index.
///
/// The attribute must be a StringAttr. Empty and duplicate names are rejected by the FuncDefOp
/// verifier.
MLIR_CAPI_EXPORTED void
llzkFunction_FuncDefOpSetResNameAttr(MlirOperation op, unsigned index, MlirAttribute attr);

/// Sets the `function.res_name` attribute for the result at the given index from a string value.
///
/// Empty and duplicate names are rejected by the FuncDefOp verifier.
MLIR_CAPI_EXPORTED void
llzkFunction_FuncDefOpSetResName(MlirOperation op, unsigned index, MlirStringRef name);

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Creates a CallOp.
LLZK_DECLARE_OP_BUILD_METHOD(
    Function, CallOp, intptr_t numResults, MlirType const *results, MlirAttribute name,
    intptr_t numOperands, MlirValue const *operands
);

/// Creates a CallOp targeting the given FuncDefOp.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCallee, MlirOperation callee, intptr_t numOperands,
    MlirValue const *operands
);

/// Creates a CallOp with affine map operands.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, WithMapOperands, intptr_t numResults, MlirType const *results,
    MlirAttribute name, LlzkAffineMapOperandsBuilder mapOperands, intptr_t numArgOperands,
    MlirValue const *argOperands
);

/// Creates a CallOp targeting the given FuncDefOp with affine map operands.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithMapOperands, MlirOperation callee,
    LlzkAffineMapOperandsBuilder mapOperands, intptr_t numArgOperands, MlirValue const *argOperands
);

/// Creates a CallOp with `templateParams` attributes.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, WithTemplateParams, intptr_t numResults, MlirType const *results,
    MlirAttribute name, intptr_t numTemplateParams, MlirAttribute const *templateParams,
    intptr_t numArgOperands, MlirValue const *argOperands
);

/// Creates a CallOp targeting the given FuncDefOp with `templateParams` attributes.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithTemplateParams, MlirOperation callee, intptr_t numTemplateParams,
    MlirAttribute const *templateParams, intptr_t numArgOperands, MlirValue const *argOperands
);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECT_FUNCTION_H
