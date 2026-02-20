//===-- OpTestBase.h - Operation unit testing infrastructure ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"

#include <gtest/gtest.h>

#include "../LLZKTestBase.h"

class OpTests : public LLZKTest {
protected:
  static constexpr auto funcNameA = "FuncA";
  static constexpr auto funcNameB = "FuncB";
  static constexpr auto structNameA = "StructA";
  static constexpr auto structNameB = "StructB";

  mlir::OwningOpRef<mlir::ModuleOp> mod;

  OpTests() : LLZKTest(), mod() {}

  void SetUp() override {
    // Create a new module for each test
    mod = llzk::createLLZKModule(&ctx, loc);
  }

  void TearDown() override {
    // Allow existing module to be erased after each test
    mod = mlir::OwningOpRef<mlir::ModuleOp>();
  }

  llzk::ModuleBuilder newEmptyExample() { return llzk::ModuleBuilder {mod.get()}; }

  llzk::ModuleBuilder newBasicFunctionsExample(
      size_t numParams = 0, std::vector<std::string_view> names = {funcNameB, funcNameA}
  ) {
    mlir::IndexType idxTy = mlir::IndexType::get(&ctx);
    llvm::SmallVector<mlir::Type> paramTypes(numParams, idxTy);
    mlir::FunctionType fTy =
        mlir::FunctionType::get(&ctx, mlir::TypeRange(paramTypes), mlir::TypeRange {idxTy});
    llzk::ModuleBuilder llzkBldr(mod.get());
    for (std::string_view n : names) {
      llzkBldr.insertFreeFunc(n, fTy);
    }
    return llzkBldr;
  }

  llzk::ModuleBuilder newStructExample(int numStructParams = -1) {
    llzk::ModuleBuilder llzkBldr(mod.get());
    llzkBldr.insertFullStruct(structNameA, numStructParams)
        .insertFullStruct(structNameB, numStructParams);
    return llzkBldr;
  }
};

template <typename TypeClass> bool verify(mlir::Operation *op, bool verifySymbolUses = false) {
  // First, call the ODS-generated function for the Op to ensure that necessary attributes exist.
  if (failed(llvm::cast<TypeClass>(op).verifyInvariants())) {
    return false;
  }
  // Second, verify all traits on the Op and call the custom verify() (if defined) via the
  // `verifyInvariants()` function in `OpDefinition.h`.
  if (failed(op->getName().verifyInvariants(op))) {
    return false;
  }
  // Finally, if applicable, call the ODS-generated `verifySymbolUses()` function.
  if (verifySymbolUses) {
    if (mlir::SymbolUserOpInterface userOp = llvm::dyn_cast<mlir::SymbolUserOpInterface>(op)) {
      mlir::SymbolTableCollection tables;
      if (failed(userOp.verifySymbolUses(tables))) {
        return false;
      }
    }
  }

  return true;
}

template <typename TypeClass> inline bool verify(TypeClass op, bool verifySymbolUses = false) {
  return verify<TypeClass>(op.getOperation(), verifySymbolUses);
}

template <typename TypeClass> inline void verifyOrDie(TypeClass op, bool verifySymbolUses = false) {
  if (!verify<TypeClass>(op.getOperation(), verifySymbolUses)) {
    std::abort();
  }
}
