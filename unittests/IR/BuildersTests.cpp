//===-- BuildersTests.cpp - Unit tests for op builders ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"

#include <gtest/gtest.h>

/* Tests for the ModuleBuilder */

using namespace llzk;

/// TODO: likely a good candidate for property-based testing.
/// A potential good option for a future date: https://github.com/emil-e/rapidcheck
class ModuleBuilderTests : public LLZKTest {
protected:
  static constexpr auto structAName = "structA";
  static constexpr auto structBName = "structB";
  static constexpr auto structCName = "structC";

  mlir::OwningOpRef<mlir::ModuleOp> mod;
  ModuleBuilder builder;

  ModuleBuilderTests() : LLZKTest(), mod(createLLZKModule(&ctx)), builder(mod.get()) {}

  void SetUp() override {
    // Create a new module and builder for each test.
    mod = createLLZKModule(&ctx);
    builder = ModuleBuilder(mod.get());
  }
};

class TemplateBuilderTests : public LLZKTest {
protected:
  mlir::OwningOpRef<mlir::ModuleOp> mod;

  TemplateBuilderTests() : LLZKTest(), mod(createLLZKModule(&ctx)) {}

  polymorphic::TemplateOp createTemplate(mlir::Location location) {
    mlir::OpBuilder builder(&ctx);

    builder.setInsertionPointToStart(mod->getBody());

    return builder.create<polymorphic::TemplateOp>(location, builder.getStringAttr("testTemplate"));
  }
};

TEST_F(ModuleBuilderTests, testModuleOpCreation) { ASSERT_NE(builder.getModule(), nullptr); }

TEST_F(ModuleBuilderTests, testStructDefInsertion) {
  builder.insertEmptyStruct(structAName);
  ASSERT_NE(builder.getStruct(structAName), nullptr);
}

TEST_F(ModuleBuilderTests, testFnInsertion) {
  builder.insertFullStruct(structAName);

  auto computeFn = builder.getComputeFn(structAName);
  ASSERT_TRUE(mlir::succeeded(computeFn));
  ASSERT_EQ(computeFn->getBody().getArguments().size(), 0);

  auto constrainFn = builder.getConstrainFn(structAName);
  ASSERT_TRUE(mlir::succeeded(constrainFn));
  ASSERT_EQ(constrainFn->getBody().getArguments().size(), 1);
}

TEST_F(ModuleBuilderTests, testConstruction) {
  builder.insertConstrainOnlyStruct(structAName)
      .insertConstrainOnlyStruct(structBName)
      .insertConstrainOnlyStruct(structCName)
      .insertConstrainCall(structAName, structBName);

  size_t numStructs = 0;
  for (auto s : builder.getModule().getOps<llzk::component::StructDefOp>()) {
    numStructs++;
    size_t numFn = 0;
    for (auto fn : s.getOps<llzk::function::FuncDefOp>()) {
      numFn++;
      ASSERT_EQ(fn.getName(), llzk::FUNC_NAME_CONSTRAIN);
    }
    ASSERT_EQ(numFn, 1);
  }
  ASSERT_EQ(numStructs, 3);

  auto aFn = builder.getConstrainFn(structAName);
  ASSERT_TRUE(mlir::succeeded(aFn));
  size_t numOps = 0;
  for ([[maybe_unused]] auto &_ : aFn->getOps()) {
    numOps++;
  }
  ASSERT_EQ(numOps, 2);
}

TEST_F(TemplateBuilderTests, testInsertAndLookupParam) {
  auto location = mlir::UnknownLoc::get(&ctx);

  auto tmpl = createTemplate(location);
  TemplateBuilder builder(tmpl);

  builder.insertParam("x", location);

  auto param = builder.getParam("x");

  ASSERT_TRUE(mlir::succeeded(param));
}

TEST_F(TemplateBuilderTests, testInsertAndLookupExpr) {
  auto location = mlir::UnknownLoc::get(&ctx);

  auto tmpl = createTemplate(location);
  TemplateBuilder builder(tmpl);

  builder.insertExpr("y", location);

  auto expr = builder.getExpr("y");

  ASSERT_TRUE(mlir::succeeded(expr));
}

TEST_F(TemplateBuilderTests, testDuplicateParamInsertion) {
  auto location = mlir::UnknownLoc::get(&ctx);

  auto tmpl = createTemplate(location);
  TemplateBuilder builder(tmpl);

  builder.insertParam("x", location);

  ASSERT_DEATH(builder.insertParam("x", location), "Duplicate TemplateParamOp insertion attempted");

  auto param = builder.getParam("x");

  ASSERT_TRUE(mlir::succeeded(param));
}

TEST_F(TemplateBuilderTests, testDuplicateExprInsertion) {
  auto location = mlir::UnknownLoc::get(&ctx);

  auto tmpl = createTemplate(location);
  TemplateBuilder builder(tmpl);

  builder.insertExpr("z", location);

  ASSERT_DEATH(builder.insertExpr("z", location), "Duplicate TemplateExprOp insertion attempted");

  auto expr = builder.getExpr("z");

  ASSERT_TRUE(mlir::succeeded(expr));
}
