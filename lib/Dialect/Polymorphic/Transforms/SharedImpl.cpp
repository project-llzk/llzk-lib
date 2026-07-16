//===-- SharedImpl.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SharedImpl.h"

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Util/SymbolHelper.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>

mlir::ConversionTarget llzk::polymorphic::detail::newBaseTarget(mlir::MLIRContext *ctx) {
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      llzk::LLZKDialect, llzk::array::ArrayDialect, llzk::boolean::BoolDialect,
      llzk::cast::CastDialect, llzk::component::StructDialect, llzk::constrain::ConstrainDialect,
      llzk::felt::FeltDialect, llzk::function::FunctionDialect, llzk::global::GlobalDialect,
      llzk::include::IncludeDialect, llzk::polymorphic::PolymorphicDialect, llzk::ram::RAMDialect,
      llzk::string::StringDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  return target;
}

llzk::array::ArrayType llzk::polymorphic::detail::flattenInstantiatedArrayType(
    llzk::array::ArrayType inputTy, mlir::Type convertedElemTy
) {
  llvm::SmallVector<mlir::Attribute> mergedDims(inputTy.getDimensionSizes());
  while (llzk::array::ArrayType nestedArrTy =
             llvm::dyn_cast<llzk::array::ArrayType>(convertedElemTy)) {
    llvm::append_range(mergedDims, nestedArrTy.getDimensionSizes());
    convertedElemTy = nestedArrTy.getElementType();
  }
  return llzk::array::ArrayType::get(convertedElemTy, mergedDims);
}

std::string llzk::polymorphic::detail::buildInstantiatedFunctionName(
    llvm::StringRef instantiatedTemplateName, llvm::StringRef functionName
) {
  return (llvm::Twine(instantiatedTemplateName) + "_" + functionName).str();
}

std::string llzk::polymorphic::detail::buildInstantiatedStructName(
    llvm::StringRef instantiatedTemplateName, llvm::StringRef structName
) {
  return (llvm::Twine(instantiatedTemplateName) + "_" + structName).str();
}

std::string llzk::polymorphic::detail::buildInstantiatedFunctionName(
    llvm::StringRef templateName, llvm::StringRef functionName,
    llvm::ArrayRef<mlir::Attribute> templateArgs
) {
  return buildInstantiatedFunctionName(
      llzk::BuildShortTypeString::from(templateName.str(), templateArgs), functionName
  );
}

std::string llzk::polymorphic::detail::buildInstantiatedStructName(
    llvm::StringRef templateName, llvm::StringRef structName,
    llvm::ArrayRef<mlir::Attribute> templateArgs
) {
  return buildInstantiatedStructName(
      llzk::BuildShortTypeString::from(templateName.str(), templateArgs), structName
  );
}

mlir::SymbolRefAttr llzk::polymorphic::detail::getInstantiatedFunctionCallee(
    mlir::SymbolRefAttr templateFunctionCallee, mlir::StringAttr instantiatedFunctionName
) {
  llvm::SmallVector<mlir::FlatSymbolRefAttr> pieces = llzk::getPieces(templateFunctionCallee);
  assert(pieces.size() >= 2 && "callee must include at least template and function names");
  pieces.pop_back();
  pieces.pop_back();
  pieces.push_back(mlir::FlatSymbolRefAttr::get(instantiatedFunctionName));
  return llzk::asSymbolRefAttr(pieces);
}

mlir::FailureOr<llzk::polymorphic::detail::FullFunctionInstantiationResult>
llzk::polymorphic::detail::getOrCreateFullFunctionInstantiation(
    mlir::ModuleOp parentModule, llzk::polymorphic::TemplateOp parentTemplate,
    llzk::function::FuncDefOp sourceFunc, mlir::SymbolRefAttr originalCallee,
    llvm::StringRef instantiatedTemplateName, mlir::SymbolTableCollection &symbolTables,
    llvm::function_ref<mlir::LogicalResult(llzk::function::FuncDefOp)> initializeClone
) {
  std::string newFuncName =
      buildInstantiatedFunctionName(instantiatedTemplateName, sourceFunc.getSymName());
  mlir::SymbolTable &moduleSymbols = symbolTables.getSymbolTable(parentModule);
  if (mlir::Operation *existing = moduleSymbols.lookup(newFuncName)) {
    auto existingFunc = llvm::dyn_cast<llzk::function::FuncDefOp>(existing);
    if (!existingFunc) {
      return mlir::failure();
    }
    return FullFunctionInstantiationResult {
        existingFunc,
        getInstantiatedFunctionCallee(originalCallee, existingFunc.getSymNameAttr()),
        /*created=*/false,
    };
  }

  llzk::function::FuncDefOp clone = sourceFunc.clone();
  clone.setSymName(newFuncName);
  moduleSymbols.insert(clone, mlir::Block::iterator(parentTemplate));
  if (mlir::failed(initializeClone(clone))) {
    clone->erase();
    return mlir::failure();
  }
  return FullFunctionInstantiationResult {
      clone,
      getInstantiatedFunctionCallee(originalCallee, clone.getSymNameAttr()),
      /*created=*/true,
  };
}
