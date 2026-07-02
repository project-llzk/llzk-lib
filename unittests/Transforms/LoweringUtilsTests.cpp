//===-- LoweringUtilsTests.cpp - Transform utility tests --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;

namespace {

class LoweringUtilsTests : public LLZKTest {};

TEST_F(LoweringUtilsTests, RejectsSingleBlockSuccessorBearingConstrainBody) {
  ctx.allowUnregisteredDialects();

  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  auto funcType = builder.getFunctionType(TypeRange {}, TypeRange {});
  auto constrainFunc = builder.create<function::FuncDefOp>(loc, "constrain", funcType);
  Block *entryBlock = constrainFunc.addEntryBlock();

  // No registered LLZK test op has successors without regions, so use a synthetic op to cover the
  // helper's successor-only rejection path without changing dialect registration.
  OperationState successorOpState(loc, "test.successor");
  successorOpState.addSuccessors(entryBlock);
  entryBlock->push_back(Operation::create(successorOpState));

  builder.setInsertionPointToEnd(entryBlock);
  builder.create<function::ReturnOp>(loc);

  EXPECT_TRUE(failed(checkFuncBodyIsStraightLine(constrainFunc, "test pass")));
}

} // namespace
