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

// Include the generated CAPI
#include "llzk/Dialect/Function/IR/Attrs.capi.cpp.inc"
#include "llzk/Dialect/Function/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Function, llzk__function, FunctionDialect)

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
    MlirAttribute name, intptr_t numMapOperands, MlirValueRange const *mapOperands,
    MlirAttribute numDimsPerMap, intptr_t numArgOperands, MlirValue const *argOperands
) {
  SmallVector<Type> resultsSto;
  SmallVector<Value> argOperandsSto;
  MapOperandsHelper<> mapOperandsHelper(numMapOperands, mapOperands);
  return wrap(
      create<CallOp>(
          builder, location, unwrapList(numResults, results, resultsSto), unwrapName(name),
          *mapOperandsHelper, unwrapDims(numDimsPerMap),
          unwrapList(numArgOperands, argOperands, argOperandsSto)
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, WithMapOperandsAndDims, intptr_t numResults, MlirType const *results,
    MlirAttribute name, intptr_t numMapOperands, MlirValueRange const *mapOperands,
    intptr_t numDimsPermMapLength, int32_t const *numDimsPerMap, intptr_t numArgOperands,
    MlirValue const *argOperands
) {
  SmallVector<Type> resultsSto;
  SmallVector<Value> argOperandsSto;
  MapOperandsHelper<> mapOperandsHelper(numMapOperands, mapOperands);
  return wrap(
      create<CallOp>(
          builder, location, unwrapList(numResults, results, resultsSto), unwrapName(name),
          *mapOperandsHelper, ArrayRef(numDimsPerMap, numDimsPermMapLength),
          unwrapList(numArgOperands, argOperands, argOperandsSto)
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithMapOperands, MlirOperation callee, intptr_t numMapOperands,
    MlirValueRange const *mapOperands, MlirAttribute numDimsPerMap, intptr_t numArgOperands,
    MlirValue const *argOperands
) {
  SmallVector<Value> argOperandsSto;
  MapOperandsHelper<> mapOperandsHelper(numMapOperands, mapOperands);
  return wrap(
      create<CallOp>(
          builder, location, unwrapCallee(callee), *mapOperandsHelper, unwrapDims(numDimsPerMap),
          unwrapList(numArgOperands, argOperands, argOperandsSto)
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithMapOperandsAndDims, MlirOperation callee, intptr_t numMapOperands,
    MlirValueRange const *mapOperands, intptr_t numDimsPermMapLength, int32_t const *numDimsPerMap,
    intptr_t numArgOperands, MlirValue const *argOperands
) {
  SmallVector<Value> argOperandsSto;
  MapOperandsHelper<> mapOperandsHelper(numMapOperands, mapOperands);
  return wrap(
      create<CallOp>(
          builder, location, unwrapCallee(callee), *mapOperandsHelper,
          ArrayRef(numDimsPerMap, numDimsPermMapLength),
          unwrapList(numArgOperands, argOperands, argOperandsSto)
      )
  );
}
