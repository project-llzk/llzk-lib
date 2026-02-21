//===-- Function.cpp - Function dialect C API implementation ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include "llzk-c/Dialect/Function.h"
#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

#include <llvm/ADT/SmallVectorExtras.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::function;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Function, llzk__function, FunctionDialect)

static NamedAttribute unwrap(MlirNamedAttribute attr) {
  return NamedAttribute(unwrap(attr.name), unwrap(attr.attribute));
}

//===----------------------------------------------------------------------===//
// FuncDefOp
//===----------------------------------------------------------------------===//

/// Creates a FuncDefOp with the given attributes and argument attributes. Each argument attribute
/// has to be a DictionaryAttr.
MlirOperation llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
    MlirLocation location, MlirStringRef name, MlirType funcType, intptr_t numAttrs,
    MlirNamedAttribute const *attrs, intptr_t numArgAttrs, MlirAttribute const *argAttrs
) {
  SmallVector<NamedAttribute> attrsSto;
  SmallVector<Attribute> argAttrsSto;
  SmallVector<DictionaryAttr> unwrappedArgAttrs =
      llvm::map_to_vector(unwrapList(numArgAttrs, argAttrs, argAttrsSto), [](auto attr) {
    return llvm::cast<DictionaryAttr>(attr);
  });
  return wrap(
      FuncDefOp::create(
          unwrap(location), unwrap(name), llvm::cast<FunctionType>(unwrap(funcType)),
          unwrapList(numAttrs, attrs, attrsSto), unwrappedArgAttrs
      )
  );
}

bool llzkOperationIsA_Function_FuncDefOp(MlirOperation op) {
  return llvm::isa<FuncDefOp>(unwrap(op));
}

bool llzkFunction_FuncDefOpGetHasAllowConstraintAttr(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).hasAllowConstraintAttr();
}

void llzkFunction_FuncDefOpSetAllowConstraintAttr(MlirOperation op, bool value) {
  unwrap_cast<FuncDefOp>(op).setAllowConstraintAttr(value);
}

bool llzkFunction_FuncDefOpGetHasAllowWitnessAttr(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).hasAllowWitnessAttr();
}

void llzkFunction_FuncDefOpSetAllowWitnessAttr(MlirOperation op, bool value) {
  unwrap_cast<FuncDefOp>(op).setAllowWitnessAttr(value);
}

bool llzkFunction_FuncDefOpGetHasAllowNonNativeFieldOpsAttr(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).hasAllowNonNativeFieldOpsAttr();
}

void llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(MlirOperation op, bool value) {
  unwrap_cast<FuncDefOp>(op).setAllowNonNativeFieldOpsAttr(value);
}

bool llzkFunction_FuncDefOpGetHasArgIsPub(MlirOperation op, unsigned argNo) {
  return unwrap_cast<FuncDefOp>(op).hasArgPublicAttr(argNo);
}

MlirAttribute llzkFunction_FuncDefOpGetFullyQualifiedName(MlirOperation op) {
  return wrap(unwrap_cast<FuncDefOp>(op).getFullyQualifiedName());
}

bool llzkFunction_FuncDefOpGetNameIsCompute(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).nameIsCompute();
}

bool llzkFunction_FuncDefOpGetNameIsConstrain(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).nameIsConstrain();
}

bool llzkFunction_FuncDefOpGetIsInStruct(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).isInStruct();
}

bool llzkFunction_FuncDefOpGetIsStructCompute(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).isStructCompute();
}

bool llzkFunction_FuncDefOpGetIsStructConstrain(MlirOperation op) {
  return unwrap_cast<FuncDefOp>(op).isStructConstrain();
}

/// Return the "self" value (i.e. the return value) from the function (which must be
/// named `FUNC_NAME_COMPUTE`).
MlirValue llzkFunction_FuncDefOpGetSelfValueFromCompute(MlirOperation op) {
  return wrap(unwrap_cast<FuncDefOp>(op).getSelfValueFromCompute());
}

/// Return the "self" value (i.e. the first parameter) from the function (which must be
/// named `FUNC_NAME_CONSTRAIN`).
MlirValue llzkFunction_FuncDefOpGetSelfValueFromConstrain(MlirOperation op) {
  return wrap(unwrap_cast<FuncDefOp>(op).getSelfValueFromConstrain());
}

/// Assuming the function is the compute function returns its StructType result.
MlirType llzkFunction_FuncDefOpGetSingleResultTypeOfCompute(MlirOperation op) {
  return wrap(unwrap_cast<FuncDefOp>(op).getSingleResultTypeOfCompute());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static auto unwrapCallee(MlirOperation op) { return llvm::cast<FuncDefOp>(unwrap(op)); }

static auto unwrapDims(MlirAttribute attr) { return llvm::cast<DenseI32ArrayAttr>(unwrap(attr)); }

static auto unwrapName(MlirAttribute attr) { return llvm::cast<SymbolRefAttr>(unwrap(attr)); }

LLZK_DEFINE_OP_BUILD_METHOD(
    Function, CallOp, intptr_t numResults, MlirType const *results, MlirAttribute name,
    intptr_t numOperands, MlirValue const *operands
) {
  SmallVector<Type> resultsSto;
  SmallVector<Value> operandsSto;
  return wrap(
      create<CallOp>(
          builder, location, unwrapList(numResults, results, resultsSto), unwrapName(name),
          unwrapList(numOperands, operands, operandsSto)
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCallee, MlirOperation callee, intptr_t numOperands,
    MlirValue const *operands
) {
  SmallVector<Value> operandsSto;
  return wrap(
      create<CallOp>(
          builder, location, unwrapCallee(callee), unwrapList(numOperands, operands, operandsSto)
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, WithMapOperands, intptr_t numResults, MlirType const *results,
    MlirAttribute name, LlzkAffineMapOperandsBuilder mapOperands, intptr_t numArgOperands,
    MlirValue const *argOperands
) {
  SmallVector<Type> resultsSto;
  SmallVector<Value> argOperandsSto;
  MapOperandsHelper<> mapOperandsHelper(mapOperands.nMapOperands, mapOperands.mapOperands);
  auto numDimsPerMap =
      llzkAffineMapOperandsBuilderGetDimsPerMapAttr(mapOperands, mlirLocationGetContext(location));
  return wrap(
      create<CallOp>(
          builder, location, unwrapList(numResults, results, resultsSto), unwrapName(name),
          *mapOperandsHelper, unwrapDims(numDimsPerMap),
          unwrapList(numArgOperands, argOperands, argOperandsSto)
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithMapOperands, MlirOperation callee,
    LlzkAffineMapOperandsBuilder mapOperands, intptr_t numArgOperands, MlirValue const *argOperands
) {
  SmallVector<Value> argOperandsSto;
  MapOperandsHelper<> mapOperandsHelper(mapOperands.nMapOperands, mapOperands.mapOperands);
  auto numDimsPerMap =
      llzkAffineMapOperandsBuilderGetDimsPerMapAttr(mapOperands, mlirLocationGetContext(location));
  return wrap(
      create<CallOp>(
          builder, location, unwrapCallee(callee), *mapOperandsHelper, unwrapDims(numDimsPerMap),
          unwrapList(numArgOperands, argOperands, argOperandsSto)
      )
  );
}

bool llzkOperationIsA_Function_CallOp(MlirOperation op) { return llvm::isa<CallOp>(unwrap(op)); }

MlirType llzkFunction_CallOpGetCalleeType(MlirOperation op) {
  return wrap(unwrap_cast<CallOp>(op).getCalleeType());
}

bool llzkFunction_CallOpGetCalleeIsCompute(MlirOperation op) {
  return unwrap_cast<CallOp>(op).calleeIsCompute();
}

bool llzkFunction_CallOpGetCalleeIsConstrain(MlirOperation op) {
  return unwrap_cast<CallOp>(op).calleeIsConstrain();
}

bool llzkFunction_CallOpGetCalleeIsStructCompute(MlirOperation op) {
  return unwrap_cast<CallOp>(op).calleeIsStructCompute();
}

bool llzkFunction_CallOpGetCalleeIsStructConstrain(MlirOperation op) {
  return unwrap_cast<CallOp>(op).calleeIsStructConstrain();
}

/// Return the "self" value (i.e. the return value) from the callee function (which must be
/// named `FUNC_NAME_COMPUTE`).
MlirValue llzkFunction_CallOpGetSelfValueFromCompute(MlirOperation op) {
  return wrap(unwrap_cast<CallOp>(op).getSelfValueFromCompute());
}

/// Return the "self" value (i.e. the first parameter) from the callee function (which must be
/// named `FUNC_NAME_CONSTRAIN`).
MlirValue llzkFunction_CallOpGetSelfValueFromConstrain(MlirOperation op) {
  return wrap(unwrap_cast<CallOp>(op).getSelfValueFromConstrain());
}

MlirType llzkFunction_CallOpGetSingleResultTypeOfCompute(MlirOperation op) {
  return wrap(unwrap_cast<CallOp>(op).getSingleResultTypeOfCompute());
}
