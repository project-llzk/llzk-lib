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

  OwningOpRef<ModuleOp> parseModuleWithUnreadMultiElementArrayWrites() {
    return parseSourceString<ModuleOp>(
        R"mlir(
module attributes {llzk.lang} {
  function.def @unused_multi_write(%value: !felt.type<"babybear">) {
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %array = array.new : !array.type<2 x !felt.type<"babybear">>
    array.write %array[%i] = %value : !array.type<2 x !felt.type<"babybear">>, !felt.type<"babybear">
    array.write %array[%j] = %value : !array.type<2 x !felt.type<"babybear">>, !felt.type<"babybear">
    function.return
  }
}
)mlir",
        ParserConfig(&ctx)
    );
  }

  OwningOpRef<ModuleOp> parseModuleWithReadAfterArrayWrite() {
    return parseSourceString<ModuleOp>(
        R"mlir(
module attributes {llzk.lang} {
  function.def @read_after_write(%value: !felt.type<"babybear">) -> !felt.type<"babybear"> {
    %i = arith.constant 0 : index
    %array = array.new : !array.type<1 x !felt.type<"babybear">>
    array.write %array[%i] = %value : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
    %loaded = array.read %array[%i] : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
    function.return %loaded : !felt.type<"babybear">
  }
}
)mlir",
        ParserConfig(&ctx)
    );
  }

  template <typename ResourceTy> void runCleanup(ModuleOp module) {
    PassManager pm(&ctx);
    pm.addPass(
        createSpecializedRemoveUnusedAllocationsPass<
            CreateArrayOp, ResourceTy, DiscardableAllocationAccessorOpInterface>()
    );
    ASSERT_TRUE(succeeded(pm.run(module)));
  }

  template <typename OpTy> static unsigned countOps(ModuleOp module) {
    unsigned count = 0;
    module.walk([&](OpTy) { ++count; });
    return count;
  }
};

TEST_F(SpecializedRemoveUnusedAllocationsTest, RequiresDiscardableAllocationResource) {
  OwningOpRef<ModuleOp> module = parseModuleWithUnreadArrayWrite();
  ASSERT_TRUE(module);

  CreateArrayOp createArray;
  module->walk([&](CreateArrayOp op) { createArray = op; });
  ASSERT_TRUE(createArray);

  EXPECT_TRUE(
      llzk::detail::hasAllocationEffectOnResource<DiscardableAllocationResource>(
          createArray.getOperation()
      )
  );
  EXPECT_FALSE(
      llzk::detail::hasAllocationEffectOnResource<SideEffects::DefaultResource>(
          createArray.getOperation()
      )
  );
}

TEST_F(SpecializedRemoveUnusedAllocationsTest, ErasesUnreadDiscardableAllocations) {
  OwningOpRef<ModuleOp> module = parseModuleWithUnreadArrayWrite();
  ASSERT_TRUE(module);

  runCleanup<DiscardableAllocationResource>(*module);

  EXPECT_EQ(countOps<CreateArrayOp>(*module), 0u);
  EXPECT_EQ(countOps<WriteArrayOp>(*module), 0u);
}

TEST_F(SpecializedRemoveUnusedAllocationsTest, ErasesUnreadMultiElementDiscardableAllocations) {
  OwningOpRef<ModuleOp> module = parseModuleWithUnreadMultiElementArrayWrites();
  ASSERT_TRUE(module);

  runCleanup<DiscardableAllocationResource>(*module);

  EXPECT_EQ(countOps<CreateArrayOp>(*module), 0u);
  EXPECT_EQ(countOps<WriteArrayOp>(*module), 0u);
}

TEST_F(SpecializedRemoveUnusedAllocationsTest, KeepsAllocationWhenAnyUserReads) {
  OwningOpRef<ModuleOp> module = parseModuleWithReadAfterArrayWrite();
  ASSERT_TRUE(module);

  runCleanup<DiscardableAllocationResource>(*module);

  EXPECT_EQ(countOps<CreateArrayOp>(*module), 1u);
  EXPECT_EQ(countOps<WriteArrayOp>(*module), 1u);
  EXPECT_EQ(countOps<ReadArrayOp>(*module), 1u);
}

} // namespace
