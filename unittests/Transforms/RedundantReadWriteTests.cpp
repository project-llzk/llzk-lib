//===-- RedundantReadWriteTests.cpp - Redundant read/write tests -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

namespace {

class ReadOnlyOp : public mlir::Op<
                       ReadOnlyOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResults,
                       mlir::MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReadOnlyOp)

  using Op::Op;

  static llvm::StringRef getOperationName() { return "test.read_only"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  void getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
    effects.emplace_back(mlir::MemoryEffects::Read::get());
  }
};

class TestDialect : public mlir::Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestDialect)

  explicit TestDialect(mlir::MLIRContext *ctxPtr)
      : Dialect(getDialectNamespace(), ctxPtr, mlir::TypeID::get<TestDialect>()) {
    addOperations<ReadOnlyOp>();
  }

  static llvm::StringRef getDialectNamespace() { return "test"; }
};

class RedundantReadWriteTests : public LLZKTest {
protected:
  RedundantReadWriteTests() { ctx.getOrLoadDialect<TestDialect>(); }
};

TEST_F(RedundantReadWriteTests, ReadOnlyOperationPreservesObservedWrite) {
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang} {
      global.def @g : !felt.type = 1
      function.def @f(%before: !felt.type, %after: !felt.type)
          attributes {function.allow_witness} {
        global.write @g = %before : !felt.type
        "test.read_only"() : () -> ()
        global.write @g = %after : !felt.type
        function.return
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createRedundantReadAndWriteEliminationPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  unsigned writes = 0;
  module->walk([&](llzk::global::GlobalWriteOp) { ++writes; });
  EXPECT_EQ(writes, 2U);
}

} // namespace
