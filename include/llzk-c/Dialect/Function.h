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

/// Creates a FuncDefOp.
static inline MlirOperation
llzkFunction_FuncDefOpCreate(MlirLocation loc, MlirStringRef name, MlirType type) {
  return llzkFunction_FuncDefOpCreateWithAttrs(loc, name, type, /*nAttrs=*/0, /*attrs=*/NULL);
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

/// Returns true if the operation is a FuncDefOp.
LLZK_DECLARE_OP_ISA(Function, FuncDefOp);

/// Returns true if the FuncDefOp has the allow_constraint attribute.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, HasAllowConstraintAttr);

/// Sets the allow_constraint attribute in the FuncDefOp operation.
MLIR_CAPI_EXPORTED void llzkFunction_FuncDefOpSetAllowConstraintAttr(MlirOperation op, bool value);

/// Returns true if the FuncDefOp has the allow_witness attribute.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, HasAllowWitnessAttr);

/// Sets the allow_witness attribute in the FuncDefOp operation.
MLIR_CAPI_EXPORTED void llzkFunction_FuncDefOpSetAllowWitnessAttr(MlirOperation op, bool value);

/// Returns true if the FuncDefOp has the allow_non_native_field_ops attribute.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, HasAllowNonNativeFieldOpsAttr);

/// Sets the allow_non_native_field_ops attribute in the FuncDefOp operation.
MLIR_CAPI_EXPORTED void
llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(MlirOperation op, bool value);

/// Returns true if the `idx`-th argument has the Pub attribute.
LLZK_DECLARE_NARY_OP_PREDICATE(Function, FuncDefOp, HasArgIsPub, unsigned arg);

/// Returns the fully qualified name of the function.
MLIR_CAPI_EXPORTED MlirAttribute llzkFunction_FuncDefOpGetFullyQualifiedName(MlirOperation op);

/// Returns true if the function's name is 'compute'.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, NameIsCompute);

/// Returns true if the function's name is 'constrain'.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, NameIsConstrain);

/// Returns true if the function's defined inside a struct.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, IsInStruct);

/// Returns true if the function is the struct's witness computation.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, IsStructCompute);

/// Returns true if the function is the struct's constrain definition.
LLZK_DECLARE_OP_PREDICATE(Function, FuncDefOp, IsStructConstrain);

/// Return the "self" value (i.e. the return value) from the function (which must be
/// named `FUNC_NAME_COMPUTE`).
MLIR_CAPI_EXPORTED MlirValue llzkFunction_FuncDefOpGetSelfValueFromCompute(MlirOperation op);

/// Return the "self" value (i.e. the first parameter) from the function (which must be
/// named `FUNC_NAME_CONSTRAIN`).
MLIR_CAPI_EXPORTED MlirValue llzkFunction_FuncDefOpGetSelfValueFromConstrain(MlirOperation op);

/// Assuming the function is the compute function, returns its StructType result.
MLIR_CAPI_EXPORTED MlirType llzkFunction_FuncDefOpGetSingleResultTypeOfCompute(MlirOperation op);

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

/// Returns true if the operation is a CallOp.
LLZK_DECLARE_OP_ISA(Function, CallOp);

/// Returns the FunctionType of the callee.
MLIR_CAPI_EXPORTED MlirType llzkFunction_CallOpGetCalleeType(MlirOperation op);

/// Returns true if the callee is named 'compute'.
LLZK_DECLARE_OP_PREDICATE(Function, CallOp, CalleeIsCompute);

/// Returns true if the callee is named 'constrain'.
LLZK_DECLARE_OP_PREDICATE(Function, CallOp, CalleeIsConstrain);

/// Returns true if the callee is the witness computation of a struct.
LLZK_DECLARE_OP_PREDICATE(Function, CallOp, CalleeIsStructCompute);

/// Returns true if the callee is the constraints definition of a struct.
LLZK_DECLARE_OP_PREDICATE(Function, CallOp, CalleeIsStructConstrain);

/// Return the "self" value (i.e. the return value) from the callee function (which must be
/// named `FUNC_NAME_COMPUTE`).
MLIR_CAPI_EXPORTED MlirValue llzkFunction_CallOpGetSelfValueFromCompute(MlirOperation op);

/// Return the "self" value (i.e. the first parameter) from the callee function (which must be
/// named `FUNC_NAME_CONSTRAIN`).
MLIR_CAPI_EXPORTED MlirValue llzkFunction_CallOpGetSelfValueFromConstrain(MlirOperation op);

/// Assuming the callee is the compute function, returns its StructType result.
MLIR_CAPI_EXPORTED MlirType llzkFunction_CallOpGetSingleResultTypeOfCompute(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif
