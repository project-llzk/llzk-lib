//===-- Function.cpp - Function dialect C API implementation ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Function.h"

#include "llzk-c/Support.h"

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

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

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithAttrsAndArgAttrs, MlirStringRef name, MlirType funcType,
    intptr_t numAttrs, MlirNamedAttribute const *attrs, intptr_t numArgAttrs,
    MlirAttribute const *argAttrs
) {
  SmallVector<NamedAttribute> attrsSto;
  SmallVector<Attribute> argAttrsSto;
  SmallVector<DictionaryAttr> unwrappedArgAttrs =
      llvm::map_to_vector(unwrapList(numArgAttrs, argAttrs, argAttrsSto), [](auto attr) {
    return llvm::cast<DictionaryAttr>(attr);
  });
  return mlirOpBuilderInsert(
      builder, wrap(
                   create<FuncDefOp>(
                       builder, location, unwrap(name), llvm::cast<FunctionType>(unwrap(funcType)),
                       unwrapList(numAttrs, attrs, attrsSto), unwrappedArgAttrs
                   )
               )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithAttrs, MlirStringRef name, MlirType funcType, intptr_t numAttrs,
    MlirNamedAttribute const *attrs
) {
  return llzkFunction_FuncDefOpBuildWithAttrsAndArgAttrs(
      builder, location, name, funcType, numAttrs, attrs, /*numArgAttrs=*/0, /*argAttrs=*/NULL
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithArgAttrs, MlirStringRef name, MlirType funcType, intptr_t numArgAttrs,
    MlirAttribute const *argAttrs
) {
  return llzkFunction_FuncDefOpBuildWithAttrsAndArgAttrs(
      builder, location, name, funcType, /*numAttrs=*/0, /*attrs=*/NULL, numArgAttrs, argAttrs
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, FuncDefOp, WithoutAttrs, MlirStringRef name, MlirType funcType
) {
  return llzkFunction_FuncDefOpBuildWithAttrs(
      builder, location, name, funcType, /*numAttrs=*/0, /*attrs=*/NULL
  );
}

MlirAttribute llzkFunction_FuncDefOpGetArgNameAttr(MlirOperation op, unsigned index) {
  std::optional<StringAttr> argNameAttr = llvm::cast<FuncDefOp>(unwrap(op)).getArgNameAttr(index);
  return wrap(argNameAttr ? Attribute(*argNameAttr) : Attribute());
}

void llzkFunction_FuncDefOpSetArgNameAttr(MlirOperation op, unsigned index, MlirAttribute attr) {
  llvm::cast<FuncDefOp>(unwrap(op)).setArgNameAttr(index, llvm::cast<StringAttr>(unwrap(attr)));
}

void llzkFunction_FuncDefOpSetArgName(MlirOperation op, unsigned index, MlirStringRef name) {
  llvm::cast<FuncDefOp>(unwrap(op)).setArgName(index, unwrap(name));
}

MlirAttribute llzkFunction_FuncDefOpGetResNameAttr(MlirOperation op, unsigned index) {
  std::optional<StringAttr> resNameAttr = llvm::cast<FuncDefOp>(unwrap(op)).getResNameAttr(index);
  return wrap(resNameAttr ? Attribute(*resNameAttr) : Attribute());
}

void llzkFunction_FuncDefOpSetResNameAttr(MlirOperation op, unsigned index, MlirAttribute attr) {
  llvm::cast<FuncDefOp>(unwrap(op)).setResNameAttr(index, llvm::cast<StringAttr>(unwrap(attr)));
}

void llzkFunction_FuncDefOpSetResName(MlirOperation op, unsigned index, MlirStringRef name) {
  llvm::cast<FuncDefOp>(unwrap(op)).setResName(index, unwrap(name));
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
  return mlirOpBuilderInsert(
      builder, wrap(
                   create<CallOp>(
                       builder, location, unwrapList(numResults, results, resultsSto),
                       unwrapName(name), unwrapList(numOperands, operands, operandsSto)
                   )
               )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCallee, MlirOperation callee, intptr_t numOperands,
    MlirValue const *operands
) {
  SmallVector<Value> operandsSto;
  return mlirOpBuilderInsert(
      builder, wrap(
                   create<CallOp>(
                       builder, location, unwrapCallee(callee),
                       unwrapList(numOperands, operands, operandsSto)
                   )
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
  return mlirOpBuilderInsert(
      builder, wrap(
                   create<CallOp>(
                       builder, location, unwrapList(numResults, results, resultsSto),
                       unwrapName(name), *mapOperandsHelper, unwrapDims(numDimsPerMap),
                       unwrapList(numArgOperands, argOperands, argOperandsSto)
                   )
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
  return mlirOpBuilderInsert(
      builder,
      wrap(
          create<CallOp>(
              builder, location, unwrapCallee(callee), *mapOperandsHelper,
              unwrapDims(numDimsPerMap), unwrapList(numArgOperands, argOperands, argOperandsSto)
          )
      )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, WithTemplateParams, intptr_t numResults, MlirType const *results,
    MlirAttribute name, intptr_t numTemplateParams, MlirAttribute const *templateParams,
    intptr_t numArgOperands, MlirValue const *argOperands
) {
  SmallVector<Type> resultsSto;
  SmallVector<Value> argOperandsSto;
  SmallVector<Attribute> templateParamsSto;
  return mlirOpBuilderInsert(
      builder, wrap(
                   create<CallOp>(
                       builder, location, unwrapList(numResults, results, resultsSto),
                       unwrapName(name), unwrapList(numArgOperands, argOperands, argOperandsSto),
                       unwrapList(numTemplateParams, templateParams, templateParamsSto)
                   )
               )
  );
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Function, CallOp, ToCalleeWithTemplateParams, MlirOperation callee, intptr_t numTemplateParams,
    MlirAttribute const *templateParams, intptr_t numArgOperands, MlirValue const *argOperands
) {
  SmallVector<Value> argOperandsSto;
  SmallVector<Attribute> templateParamsSto;
  return mlirOpBuilderInsert(
      builder, wrap(
                   create<CallOp>(
                       builder, location, unwrapCallee(callee),
                       unwrapList(numArgOperands, argOperands, argOperandsSto),
                       unwrapList(numTemplateParams, templateParams, templateParamsSto)
                   )
               )
  );
}
