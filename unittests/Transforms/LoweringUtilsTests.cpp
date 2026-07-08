//===-- LoweringUtilsTests.cpp - Transform utility tests --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::pod;

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

static ReadPodOp findArrayOfPodFieldRead(ModuleOp module) {
  ReadPodOp readOp;
  WalkResult walkResult = module.walk([&](ReadPodOp candidate) {
    auto arrTy = dyn_cast<ArrayType>(candidate.getType());
    if (!arrTy || !isa<PodType>(arrTy.getElementType())) {
      return WalkResult::advance();
    }
    readOp = candidate;
    return WalkResult::interrupt();
  });
  EXPECT_TRUE(walkResult.wasInterrupted());
  return readOp;
}

TEST_F(LoweringUtilsTests, DetectsLoopCarriedFreshArrayFieldRead) {
  constexpr llvm::StringLiteral source = R"mlir(
    !Item = !pod.type<[@x: index]>
    !Outer = !pod.type<[@items: !array.type<2 x !Item>]>
    module attributes {llzk.lang} {
      function.def @test() {
        %state = pod.new : !Outer
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        scf.for %iv = %c0 to %c2 step %c1 {
          %items = pod.read %state[@items] : !Outer, !array.type<2 x !Item>
          %elem = array.read %items[%c0] : !array.type<2 x !Item>, !Item
          %x = pod.read %elem[@x] : !Item, index
          %next = pod.new : !Item
          pod.write %next[@x] = %x : !Item, index
          array.write %items[%c0] = %next : !array.type<2 x !Item>, !Item
          pod.write %state[@items] = %items : !Outer, !array.type<2 x !Item>
        }
        function.return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(module) << "failed to parse test module";

  ReadPodOp readOp = findArrayOfPodFieldRead(*module);
  ASSERT_TRUE(readOp);

  Operation *anchor = llzk::pod::detail::findNearestLoopCarriedPodAccess(readOp);
  ASSERT_NE(anchor, nullptr);
  EXPECT_TRUE(isa<scf::ForOp>(anchor));
}

TEST_F(LoweringUtilsTests, IgnoresLoopLocalFreshArrayFieldRead) {
  constexpr llvm::StringLiteral source = R"mlir(
    !Item = !pod.type<[@x: index]>
    !Outer = !pod.type<[@items: !array.type<2 x !Item>]>
    module attributes {llzk.lang} {
      function.def @test() {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        scf.for %iv = %c0 to %c2 step %c1 {
          %state = pod.new : !Outer
          %items = pod.read %state[@items] : !Outer, !array.type<2 x !Item>
          %elem = array.read %items[%c0] : !array.type<2 x !Item>, !Item
          %x = pod.read %elem[@x] : !Item, index
          %next = pod.new : !Item
          pod.write %next[@x] = %x : !Item, index
          array.write %items[%c0] = %next : !array.type<2 x !Item>, !Item
          pod.write %state[@items] = %items : !Outer, !array.type<2 x !Item>
        }
        function.return
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(module) << "failed to parse test module";

  ReadPodOp readOp = findArrayOfPodFieldRead(*module);
  ASSERT_TRUE(readOp);

  EXPECT_EQ(llzk::pod::detail::findNearestLoopCarriedPodAccess(readOp), nullptr);
}

TEST_F(LoweringUtilsTests, AppendValuesWithExactTypesDoesNotLeakPartialMatches) {
  OpBuilder builder(&ctx);
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  auto funcType = builder.getFunctionType(TypeRange {}, TypeRange {});
  auto func = builder.create<function::FuncDefOp>(loc, "test", funcType);
  Block *entryBlock = func.addEntryBlock();
  entryBlock->addArguments(
      {builder.getIndexType(), builder.getI1Type(), builder.getIndexType()},
      SmallVector<Location>(3, loc)
  );
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<function::ReturnOp>(loc);

  SmallVector<Value> candidateValues {entryBlock->getArgument(0), entryBlock->getArgument(1)};
  SmallVector<Type> expectedTypes {builder.getIndexType(), builder.getIndexType()};
  SmallVector<Value> collectedValues {entryBlock->getArgument(2)};

  EXPECT_FALSE(llzk::pod::detail::appendValuesWithExactTypes(
      candidateValues, expectedTypes, collectedValues
  ));
  ASSERT_EQ(collectedValues.size(), 1u);
  EXPECT_EQ(collectedValues.front(), entryBlock->getArgument(2));
}

} // namespace
