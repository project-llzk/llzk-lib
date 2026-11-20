//===-- LoweringUtilsTests.cpp - Transform utility tests --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LoweringUtils.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/raw_ostream.h>

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
  auto constrainFunc = function::FuncDefOp::create(builder, loc, "constrain", funcType);
  Block *entryBlock = constrainFunc.addEntryBlock();

  // No registered LLZK test op has successors without regions, so use a synthetic op to cover the
  // helper's successor-only rejection path without changing dialect registration.
  OperationState successorOpState(loc, "test.successor");
  successorOpState.addSuccessors(entryBlock);
  entryBlock->push_back(Operation::create(successorOpState));

  builder.setInsertionPointToEnd(entryBlock);
  function::ReturnOp::create(builder, loc);

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
  auto func = function::FuncDefOp::create(builder, loc, "test", funcType);
  Block *entryBlock = func.addEntryBlock();
  entryBlock->addArguments(
      {builder.getIndexType(), builder.getI1Type(), builder.getIndexType()},
      SmallVector<Location>(3, loc)
  );
  builder.setInsertionPointToEnd(entryBlock);
  function::ReturnOp::create(builder, loc);

  SmallVector<Value> candidateValues {entryBlock->getArgument(0), entryBlock->getArgument(1)};
  SmallVector<Type> expectedTypes {builder.getIndexType(), builder.getIndexType()};
  SmallVector<Value> collectedValues {entryBlock->getArgument(2)};

  EXPECT_FALSE(
      llzk::pod::detail::appendValuesWithExactTypes(candidateValues, expectedTypes, collectedValues)
  );
  ASSERT_EQ(collectedValues.size(), 1u);
  EXPECT_EQ(collectedValues.front(), entryBlock->getArgument(2));
}

// Exercise defensive reconstruction on intermediate IR: normal constrain verification
// already forbids witness-only member writes before the lowering pass runs.
TEST_F(LoweringUtilsTests, RejectsMemberReplacementDuringAuxiliaryReconstruction) {
  constexpr llvm::StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ReplacedArray {
    struct.member @data : !array.type<1 x !felt.type>
    function.def @compute(%replacement: !array.type<1 x !felt.type>, %x: !felt.type) -> !struct.type<@ReplacedArray> {
      %self = struct.new : <@ReplacedArray>
      function.return %self : !struct.type<@ReplacedArray>
    }
    function.def @constrain(%self: !struct.type<@ReplacedArray>, %replacement: !array.type<1 x !felt.type>, %x: !felt.type) {
      struct.writem %self[@data] = %replacement : <@ReplacedArray>, !array.type<1 x !felt.type>
      %data = struct.readm %self[@data] : <@ReplacedArray>, !array.type<1 x !felt.type>
      %idx = arith.constant 0 : index
      %value = array.read %data[%idx] : !array.type<1 x !felt.type>, !felt.type
      %square = felt.mul %value, %value : !felt.type, !felt.type
      %fourth = felt.mul %square, %square : !felt.type, !felt.type
      constrain.eq %x, %fourth : !felt.type
      function.return
    }
  }
}

  )mlir";
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(source, ParserConfig(&ctx, /*verifyAfterParse=*/false));
  ASSERT_TRUE(module);

  function::FuncDefOp compute;
  ReadArrayOp read;
  module->walk([&](function::FuncDefOp func) {
    if (func.isStructCompute()) {
      compute = func;
    }
  });
  module->walk([&](ReadArrayOp op) { read = op; });
  ASSERT_TRUE(compute);
  ASSERT_TRUE(read);

  std::string message;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diagnostic) {
    llvm::raw_string_ostream stream(message);
    diagnostic.print(stream);
    return success();
  });
  OpBuilder builder(compute.getBody().front().getTerminator());
  llvm::DenseMap<Value, Value> memo;
  EXPECT_FALSE(rebuildExprInCompute(read.getResult(), compute, builder, memo));
  EXPECT_NE(message.find("aggregate mutations cannot be replayed"), std::string::npos);
}

} // namespace
