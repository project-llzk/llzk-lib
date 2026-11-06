//===-- CastEffectTests.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Cast/IR/Ops.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser/Parser.h>

#include <string>

namespace {

class CastEffectTests : public LLZKTest {};

TEST_F(CastEffectTests, OverflowModeControlsEliminationAndSpeculation) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang} {
      function.def @casts(%i: index, %f: !felt.type<"bn254">)
          attributes {function.allow_non_native_field_ops} {
        %a = cast.tofelt %i : index, !felt.type<"babybear">
        %b = cast.tofelt sat %i : index, !felt.type<"babybear">
        %c = cast.tofelt wrap %i : index, !felt.type<"babybear">
        %d = cast.tofelt trunc %i : index, !felt.type<"babybear">
        %e = cast.toindex %f : !felt.type<"bn254">
        %f0 = cast.toindex sat %f : !felt.type<"bn254">
        %g = cast.toindex wrap %f : !felt.type<"bn254">
        %h = cast.toindex trunc %f : !felt.type<"bn254">
        function.return
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  unsigned count = 0;
  module->walk([&](mlir::Operation *op) {
    llzk::cast::OverflowSemantics mode;
    if (auto toFelt = llvm::dyn_cast<llzk::cast::IntToFeltOp>(op)) {
      mode = toFelt.getOverflow();
    } else if (auto toIndex = llvm::dyn_cast<llzk::cast::FeltToIndexOp>(op)) {
      mode = toIndex.getOverflow();
    } else {
      return;
    }
    ++count;
    SCOPED_TRACE(
        std::string(op->getName().getStringRef()) + ": " +
        llzk::cast::stringifyOverflowSemantics(mode).str()
    );
    bool canEliminate = mode != llzk::cast::OverflowSemantics::ASSERT;
    EXPECT_EQ(mlir::isMemoryEffectFree(op), canEliminate);
    EXPECT_EQ(mlir::isSpeculatable(op), canEliminate);
    EXPECT_EQ(mlir::isPure(op), canEliminate);
    EXPECT_EQ(mlir::isOpTriviallyDead(op), canEliminate);
  });
  EXPECT_EQ(count, 8U);
}

// Dynamic booleans fit in every field, including one that has not been selected yet.
TEST_F(CastEffectTests, BooleanCastsAreAlwaysPure) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang} {
      function.def @casts(%flag: i1) {
        %a = cast.tofelt %flag : i1, !felt.type<"babybear">
        %b = cast.tofelt assert %flag : i1, !felt.type<"babybear">
        %c = cast.tofelt sat %flag : i1, !felt.type<"babybear">
        %d = cast.tofelt wrap %flag : i1, !felt.type<"babybear">
        %e = cast.tofelt trunc %flag : i1, !felt.type<"babybear">
        %f = cast.tofelt %flag : i1
        %g = cast.tofelt assert %flag : i1
        %h = cast.tofelt sat %flag : i1
        %i = cast.tofelt wrap %flag : i1
        %j = cast.tofelt trunc %flag : i1
        function.return
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  unsigned count = 0;
  module->walk([&](llzk::cast::IntToFeltOp op) {
    ++count;
    SCOPED_TRACE(count);
    EXPECT_TRUE(mlir::isMemoryEffectFree(op));
    EXPECT_TRUE(mlir::isSpeculatable(op));
    EXPECT_TRUE(mlir::isPure(op));
    EXPECT_TRUE(mlir::isOpTriviallyDead(op));
  });
  EXPECT_EQ(count, 10U);
}

TEST_F(CastEffectTests, SourceFieldRangeControlsEliminationAndSpeculation) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang, llzk.fields = [
      #felt.field<"cast_fits_index", 9223372036854775783>,
      #felt.field<"cast_exceeds_index", 9223372036854775837>
    ]} {
      function.def @casts(%bb: !felt.type<"babybear">,
                          %fits: !felt.type<"cast_fits_index">,
                          %exceeds: !felt.type<"cast_exceeds_index">,
                          %goldilocks: !felt.type<"goldilocks">,
                          %bn: !felt.type<"bn254">, %unknown: !felt.type)
          attributes {function.allow_non_native_field_ops} {
        %a = cast.toindex %bb : !felt.type<"babybear"> {test.pure}
        %b = cast.toindex assert %bb : !felt.type<"babybear"> {test.pure}
        %c = cast.toindex %fits : !felt.type<"cast_fits_index"> {test.pure}
        %d = cast.toindex assert %fits : !felt.type<"cast_fits_index"> {test.pure}
        %e = cast.toindex %exceeds : !felt.type<"cast_exceeds_index">
        %f = cast.toindex assert %exceeds : !felt.type<"cast_exceeds_index">
        %g = cast.toindex %goldilocks : !felt.type<"goldilocks">
        %h = cast.toindex assert %goldilocks : !felt.type<"goldilocks">
        %i = cast.toindex %bn : !felt.type<"bn254">
        %j = cast.toindex assert %bn : !felt.type<"bn254">
        %k = cast.toindex %unknown
        %l = cast.toindex assert %unknown
        function.return
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  unsigned count = 0;
  module->walk([&](llzk::cast::FeltToIndexOp op) {
    ++count;
    SCOPED_TRACE(count);
    bool canEliminate = op->hasAttr("test.pure");
    EXPECT_EQ(mlir::isMemoryEffectFree(op), canEliminate);
    EXPECT_EQ(mlir::isSpeculatable(op), canEliminate);
    EXPECT_EQ(mlir::isPure(op), canEliminate);
    EXPECT_EQ(mlir::isOpTriviallyDead(op), canEliminate);
  });
  EXPECT_EQ(count, 12U);
}

} // namespace
