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

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Function, llzk__function);

//===----------------------------------------------------------------------===//
// FuncDefOp
//===----------------------------------------------------------------------===//

/// Creates a FuncDefOp with the given attributes and argument attributes. Each argument attribute
/// has to be a DictionaryAttr.
MLIR_CAPI_EXPORTED MlirOperation llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
    MlirLocation loc, MlirStringRef name, MlirType type, intptr_t nAttrs,
    MlirNamedAttribute const *attrs, intptr_t nArgAttrs, MlirAttribute const *argAttrs
);

/// Creates a FuncDefOp with the given attributes.
static inline MlirOperation llzkFunction_FuncDefOpCreateWithAttrs(
    MlirLocation loc, MlirStringRef name, MlirType type, intptr_t nAttrs,
    MlirNamedAttribute const *attrs
) {
  return llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
      loc, name, type, nAttrs, attrs, /*nArgAttrs=*/0, /*argAttrs=*/NULL
  );
}

/// Creates a FuncDefOp with the given argument attributes. Each argument attribute has to be a
/// DictionaryAttr.
static inline MlirOperation llzkFunction_FuncDefOpCreateWithArgAttrs(
    MlirLocation loc, MlirStringRef name, MlirType type, intptr_t nArgAttrs,
    MlirAttribute const *argAttrs
) {
  return llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
      loc, name, type, /*nAttrs=*/0, /*attrs=*/NULL, nArgAttrs, argAttrs
  );
}

/// Creates a FuncDefOp.
static inline MlirOperation
llzkFunction_FuncDefOpCreateWithoutAttrs(MlirLocation loc, MlirStringRef name, MlirType type) {
  return llzkFunction_FuncDefOpCreateWithAttrs(loc, name, type, /*nAttrs=*/0, /*attrs=*/NULL);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Creates a CallOp.
LLZK_DECLARE_OP_BUILD_METHOD(
    Function, CallOp, intptr_t numResults, MlirType const *results, MlirAttribute name,
    intptr_t numOperands, MlirValue const *operands
);

/// Creates a CallOp that calls the given FuncDefOp.
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

/// Creates a CallOp with affine map operands to the given FuncDefOp.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithMapOperands, MlirOperation callee,
    LlzkAffineMapOperandsBuilder mapOperands, intptr_t numArgOperands, MlirValue const *argOperands
);

#ifdef __cplusplus
}
#endif

#endif
