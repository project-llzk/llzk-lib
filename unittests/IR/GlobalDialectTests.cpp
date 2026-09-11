//===-- GlobalDialectTests.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"

#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser/Parser.h>

namespace {

class GlobalDialectTests : public LLZKTest {};

TEST_F(GlobalDialectTests, AccessEffects) {
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang} {
      global.def const @constant : !felt.type = 1
      global.def @mutable : !felt.type = 2
      function.def @f(%value: !felt.type) attributes {function.allow_witness} {
        %constant = global.read const @constant : !felt.type
        %mutable = global.read @mutable : !felt.type
        global.write @mutable = %value : !felt.type
        function.return
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  auto func = *module->getOps<llzk::function::FuncDefOp>().begin();
  auto reads = func.getOps<llzk::global::GlobalReadOp>();
  auto readIt = reads.begin();
  auto constantRead = *readIt++;
  auto mutableRead = *readIt;
  auto write = *func.getOps<llzk::global::GlobalWriteOp>().begin();

  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  constantRead.getEffects(effects);
  EXPECT_TRUE(effects.empty());

  mutableRead.getEffects(effects);
  ASSERT_EQ(effects.size(), 1U);
  EXPECT_TRUE(llvm::isa<mlir::MemoryEffects::Read>(effects.front().getEffect()));
  EXPECT_TRUE(llvm::isa<llzk::global::GlobalMemoryResource>(effects.front().getResource()));

  effects.clear();
  write.getEffects(effects);
  ASSERT_EQ(effects.size(), 1U);
  EXPECT_TRUE(llvm::isa<mlir::MemoryEffects::Write>(effects.front().getEffect()));
  EXPECT_TRUE(llvm::isa<llzk::global::GlobalMemoryResource>(effects.front().getResource()));
}

} // namespace
