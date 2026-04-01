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
  static constexpr auto templateName = "ExampleTemplate";

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

  template <typename Derived>
  void insertFreeFuncsWithIndexArgs(
      llzk::ModuleLikeBuilder<Derived> &bldr, size_t numArgs,
      const std::vector<std::string_view> &names
  ) {
    mlir::IndexType idxTy = mlir::IndexType::get(&ctx);
    llvm::SmallVector<mlir::Type> argTypes(numArgs, idxTy);
    mlir::FunctionType fTy =
        mlir::FunctionType::get(&ctx, mlir::TypeRange(argTypes), mlir::TypeRange {idxTy});
    for (std::string_view n : names) {
      bldr.insertFreeFunc(n, fTy);
    }
  }

  // Create 2 free functions with index-type arguments.
  llzk::ModuleBuilder newBasicFunctionsExample(
      size_t numArgs = 0, const std::vector<std::string_view> &names = {funcNameB, funcNameA}
  ) {
    llzk::ModuleBuilder modBldr(mod.get());
    insertFreeFuncsWithIndexArgs(modBldr, numArgs, names);
    return modBldr;
  }

  // Functions are nested in the template `templateName`. Returns both the ModuleBuilder (which
  // must stay alive to keep the TemplateBuilder valid) and a pointer to the TemplateBuilder.
  std::pair<llzk::ModuleBuilder, llzk::TemplateBuilder *> newTemplateFunctionsExample(
      unsigned numParams, size_t numArgs = 0,
      const std::vector<std::string_view> &names = {funcNameB, funcNameA}
  ) {
    llzk::ModuleBuilder modBldr(mod.get());
    // Note: `numParams==0` creates TemplateOp with no TemplateParamOp
    auto r = modBldr.insertTemplate(templateName, numParams).getTemplate(templateName);
    assert(mlir::succeeded(r));
    llzk::TemplateBuilder *tmplBldr = r.value();
    insertFreeFuncsWithIndexArgs(*tmplBldr, numArgs, names);
    return {std::move(modBldr), tmplBldr};
  }

  // Create 2 structs with compute and constrain functions.
  llzk::ModuleBuilder newStructExample() {
    llzk::ModuleBuilder modBldr(mod.get());
    modBldr.insertFullStruct(structNameA).insertFullStruct(structNameB);
    return modBldr;
  }

  // Structs are nested in the template `templateName`. Returns both the ModuleBuilder (which
  // must stay alive to keep the TemplateBuilder valid) and a pointer to the TemplateBuilder.
  std::pair<llzk::ModuleBuilder, llzk::TemplateBuilder *>
  newTemplateStructExample(unsigned numParams) {
    llzk::ModuleBuilder modBldr(mod.get());
    // Note: `numParams==0` creates TemplateOp with no TemplateParamOp
    auto r = modBldr.insertTemplate(templateName, numParams).getTemplate(templateName);
    assert(mlir::succeeded(r));
    llzk::TemplateBuilder *tmplBldr = r.value();
    tmplBldr->insertFullStruct(structNameA).insertFullStruct(structNameB);
    return {std::move(modBldr), tmplBldr};
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
