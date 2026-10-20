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

} // namespace
