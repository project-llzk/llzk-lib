//===-- SpecializedMemoryPassesTests.cpp - Memory pass tests ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;

namespace {

struct NonDiscardableAllocationResource
    : public SideEffects::Resource::Base<NonDiscardableAllocationResource> {
  StringRef getName() final { return "NonDiscardableAllocation"; }
};

class SpecializedRemoveUnusedAllocationsTest : public LLZKTest {
protected:
  OwningOpRef<ModuleOp> parseModuleWithUnreadArrayWrite() {
    return parseSourceString<ModuleOp>(
        R"mlir(
module attributes {llzk.lang} {
  function.def @unused_write(%value: !felt.type<"babybear">) {
    %i = arith.constant 0 : index
    %array = array.new %value : !array.type<1 x !felt.type<"babybear">>
    array.write %array[%i] = %value : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
    function.return
  }
}
)mlir",
        ParserConfig(&ctx)
    );
  }

  template <typename ResourceTy> void runCleanup(ModuleOp module) {
    PassManager pm(&ctx);
    pm.addPass(createSpecializedRemoveUnusedAllocationsPass<CreateArrayOp, ResourceTy>());
    ASSERT_TRUE(succeeded(pm.run(module)));
  }

  template <typename OpTy> static unsigned countOps(ModuleOp module) {
    unsigned count = 0;
    module.walk([&](OpTy) { ++count; });
    return count;
  }
};

TEST_F(SpecializedRemoveUnusedAllocationsTest, RequiresMatchingAllocationResource) {
  OwningOpRef<ModuleOp> module = parseModuleWithUnreadArrayWrite();
  ASSERT_TRUE(module);

  runCleanup<NonDiscardableAllocationResource>(*module);

  EXPECT_EQ(countOps<CreateArrayOp>(*module), 1u);
  EXPECT_EQ(countOps<WriteArrayOp>(*module), 1u);
}

TEST_F(SpecializedRemoveUnusedAllocationsTest, ErasesUnreadDiscardableAllocations) {
  OwningOpRef<ModuleOp> module = parseModuleWithUnreadArrayWrite();
  ASSERT_TRUE(module);

  runCleanup<DiscardableAllocationResource>(*module);

  EXPECT_EQ(countOps<CreateArrayOp>(*module), 0u);
  EXPECT_EQ(countOps<WriteArrayOp>(*module), 0u);
}

} // namespace
