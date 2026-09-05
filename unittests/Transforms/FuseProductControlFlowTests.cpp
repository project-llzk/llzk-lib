//===-- FuseProductControlFlowTests.cpp - Control-flow fusion tests -*- C++ -*-===//
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
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/RAM/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <llvm/ADT/SmallVector.h>

namespace {

class FuseProductControlFlowTests : public LLZKTest {};

TEST_F(FuseProductControlFlowTests, HoistedSignalMemberReadsPreserveSourceOrder) {
  // Distinct signal members make both source order and placement before the fused if observable.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @left : !felt.type {signal}
        struct.member @right : !felt.type {signal}

        function.def @product(%condition: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>

          %left, %right = scf.if %condition -> (!felt.type, !felt.type) {
            %zero = felt.const 0
            %one = felt.const 1
            scf.yield %zero, %one : !felt.type, !felt.type
          } else {
            %two = felt.const 2
            %three = felt.const 3
            scf.yield %two, %three : !felt.type, !felt.type
          } {product_source = "compute"}

          struct.writem %self[@left] = %left : <@A>, !felt.type
          struct.writem %self[@right] = %right : <@A>, !felt.type
          %left_read = struct.readm %self[@left] : <@A>, !felt.type {
            product_source = "constrain"
          }
          %right_read = struct.readm %self[@right] : <@A>, !felt.type {
            product_source = "constrain"
          }

          scf.if %condition {
            %expected_left = felt.const 0
            %expected_right = felt.const 1
            constrain.eq %left_read, %expected_left : !felt.type, !felt.type
            constrain.eq %right_read, %expected_right : !felt.type, !felt.type
          } else {
            %expected_left = felt.const 2
            %expected_right = felt.const 3
            constrain.eq %left_read, %expected_left : !felt.type, !felt.type
            constrain.eq %right_read, %expected_right : !felt.type, !felt.type
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  llvm::SmallVector<llzk::component::MemberReadOp> reads;
  mlir::scf::IfOp fusedIf;
  for (mlir::Operation &op : product.getBody().front()) {
    if (auto read = llvm::dyn_cast<llzk::component::MemberReadOp>(&op)) {
      reads.push_back(read);
    } else if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(&op)) {
      if (auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source");
          source && source.getValue() == "fused") {
        fusedIf = ifOp;
      }
    }
  }

  ASSERT_EQ(reads.size(), 2U);
  ASSERT_TRUE(fusedIf);
  EXPECT_EQ(reads[0].getMemberName(), "left");
  EXPECT_EQ(reads[1].getMemberName(), "right");
  EXPECT_TRUE(reads[0]->isBeforeInBlock(reads[1]));
  EXPECT_TRUE(reads[0]->isBeforeInBlock(fusedIf));
  EXPECT_TRUE(reads[1]->isBeforeInBlock(fusedIf));
}

TEST_F(FuseProductControlFlowTests, NonSignalMemberReadPreventsFusion) {
  // A non-signal member is an intermediate expression, so its read must stay after the write.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product(%condition: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>

          %value = scf.if %condition -> !felt.type {
            %zero = felt.const 0
            scf.yield %zero : !felt.type
          } else {
            %one = felt.const 1
            scf.yield %one : !felt.type
          } {product_source = "compute"}

          struct.writem %self[@value] = %value : <@A>, !felt.type
          %value_read = struct.readm %self[@value] : <@A>, !felt.type {
            product_source = "constrain"
          }

          scf.if %condition {
            %expected = felt.const 0
            constrain.eq %value_read, %expected : !felt.type, !felt.type
          } else {
            %expected = felt.const 1
            constrain.eq %value_read, %expected : !felt.type, !felt.type
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  llzk::component::MemberWriteOp write;
  llzk::component::MemberReadOp read;
  mlir::scf::IfOp fusedIf;
  for (mlir::Operation &op : product.getBody().front()) {
    if (auto writeOp = llvm::dyn_cast<llzk::component::MemberWriteOp>(&op)) {
      write = writeOp;
    } else if (auto readOp = llvm::dyn_cast<llzk::component::MemberReadOp>(&op)) {
      read = readOp;
    } else if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(&op)) {
      if (auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source");
          source && source.getValue() == "fused") {
        fusedIf = ifOp;
      }
    }
  }

  ASSERT_TRUE(write);
  ASSERT_TRUE(read);
  EXPECT_FALSE(fusedIf);
  EXPECT_TRUE(write->isBeforeInBlock(read));
}

TEST_F(FuseProductControlFlowTests, UnmarkedSignalMemberReadPreventsFusion) {
  // A signal read without a constrain source marker is not eligible for hoisting across its write.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type {signal}

        function.def @product(%condition: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>

          %value = scf.if %condition -> !felt.type {
            %zero = felt.const 0
            scf.yield %zero : !felt.type
          } else {
            %one = felt.const 1
            scf.yield %one : !felt.type
          } {product_source = "compute"}

          struct.writem %self[@value] = %value : <@A>, !felt.type
          %value_read = struct.readm %self[@value] : <@A>, !felt.type

          scf.if %condition {
            %expected = felt.const 0
            constrain.eq %value_read, %expected : !felt.type, !felt.type
          } else {
            %expected = felt.const 1
            constrain.eq %value_read, %expected : !felt.type, !felt.type
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  llzk::component::MemberWriteOp write;
  llzk::component::MemberReadOp read;
  mlir::scf::IfOp fusedIf;
  for (mlir::Operation &op : product.getBody().front()) {
    if (auto writeOp = llvm::dyn_cast<llzk::component::MemberWriteOp>(&op)) {
      write = writeOp;
    } else if (auto readOp = llvm::dyn_cast<llzk::component::MemberReadOp>(&op)) {
      read = readOp;
    } else if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(&op)) {
      if (auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source");
          source && source.getValue() == "fused") {
        fusedIf = ifOp;
      }
    }
  }

  ASSERT_TRUE(write);
  ASSERT_TRUE(read);
  EXPECT_FALSE(fusedIf);
  EXPECT_TRUE(write->isBeforeInBlock(read));
}

TEST_F(FuseProductControlFlowTests, RepeatedMemberWritesPreventReadHoisting) {
  // A hoisted signal read must have one matching write. With repeated writes, the pass cannot
  // preserve which written value the original read observed.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type {signal}

        function.def @product(%condition: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>

          %first, %second = scf.if %condition -> (!felt.type, !felt.type) {
            %zero = felt.const 0
            %two = felt.const 2
            scf.yield %zero, %two : !felt.type, !felt.type
          } else {
            %one = felt.const 1
            %three = felt.const 3
            scf.yield %one, %three : !felt.type, !felt.type
          } {product_source = "compute"}

          struct.writem %self[@value] = %first : <@A>, !felt.type {
            product_source = "compute"
          }
          %observed = struct.readm %self[@value] : <@A>, !felt.type {
            product_source = "constrain"
          }
          struct.writem %self[@value] = %second : <@A>, !felt.type {
            product_source = "compute"
          }

          scf.if %condition {
            %expected = felt.const 0
            constrain.eq %observed, %expected : !felt.type, !felt.type
          } else {
            %expected = felt.const 1
            constrain.eq %observed, %expected : !felt.type, !felt.type
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  llvm::SmallVector<llzk::component::MemberWriteOp> writes;
  llzk::component::MemberReadOp read;
  unsigned computeIfs = 0;
  unsigned constrainIfs = 0;
  unsigned fusedIfs = 0;
  product.walk([&](llzk::component::MemberWriteOp write) { writes.push_back(write); });
  product.walk([&](llzk::component::MemberReadOp memberRead) { read = memberRead; });
  product.walk([&](mlir::scf::IfOp ifOp) {
    if (auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source"); source) {
      if (source.getValue() == "compute") {
        ++computeIfs;
      } else if (source.getValue() == "constrain") {
        ++constrainIfs;
      } else if (source.getValue() == "fused") {
        ++fusedIfs;
      }
    }
  });

  ASSERT_EQ(writes.size(), 2U);
  ASSERT_TRUE(read);
  EXPECT_NE(writes[0].getVal(), writes[1].getVal());
  EXPECT_EQ(computeIfs, 1U);
  EXPECT_EQ(constrainIfs, 1U);
  EXPECT_EQ(fusedIfs, 0U);
  EXPECT_TRUE(writes[0]->isBeforeInBlock(read));
  EXPECT_TRUE(read->isBeforeInBlock(writes[1]));
}

TEST_F(FuseProductControlFlowTests, ComputeIfEffectsPreventReadHoisting) {
  // A nested compute-side member write is crossed by read hoisting, so the pair must remain
  // separate even though the direct write/read interval is otherwise valid.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type {signal}

        function.def @product(%condition: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>

          %result = scf.if %condition -> !felt.type {
            %zero = felt.const 0
            struct.writem %self[@value] = %zero : <@A>, !felt.type
            scf.yield %zero : !felt.type
          } else {
            %one = felt.const 1
            struct.writem %self[@value] = %one : <@A>, !felt.type
            scf.yield %one : !felt.type
          } {product_source = "compute"}

          struct.writem %self[@value] = %result : <@A>, !felt.type {
            product_source = "compute"
          }
          %observed = struct.readm %self[@value] : <@A>, !felt.type {
            product_source = "constrain"
          }

          scf.if %condition {
            %expected = felt.const 0
            constrain.eq %observed, %expected : !felt.type, !felt.type
          } else {
            %expected = felt.const 1
            constrain.eq %observed, %expected : !felt.type, !felt.type
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  mlir::scf::IfOp computeIf;
  mlir::scf::IfOp constrainIf;
  mlir::scf::IfOp fusedIf;
  llzk::component::MemberWriteOp delayedWrite;
  llzk::component::MemberReadOp read;
  for (mlir::Operation &op : product.getBody().front()) {
    if (auto write = llvm::dyn_cast<llzk::component::MemberWriteOp>(&op)) {
      delayedWrite = write;
    } else if (auto readOp = llvm::dyn_cast<llzk::component::MemberReadOp>(&op)) {
      read = readOp;
    } else if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(&op)) {
      auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source");
      if (source && source.getValue() == "compute") {
        computeIf = ifOp;
      } else if (source && source.getValue() == "constrain") {
        constrainIf = ifOp;
      } else if (source && source.getValue() == "fused") {
        fusedIf = ifOp;
      }
    }
  }

  ASSERT_TRUE(computeIf);
  ASSERT_TRUE(constrainIf);
  ASSERT_TRUE(delayedWrite);
  ASSERT_TRUE(read);
  EXPECT_FALSE(fusedIf);
  EXPECT_TRUE(computeIf->isBeforeInBlock(delayedWrite));
  EXPECT_TRUE(computeIf->isBeforeInBlock(read));
  EXPECT_TRUE(delayedWrite->isBeforeInBlock(read));
  EXPECT_TRUE(read->isBeforeInBlock(constrainIf));
}

TEST_F(FuseProductControlFlowTests, UnsafeOperationsBetweenIfsPreventFusion) {
  // A generic intervening definition would lose dominance if the conditionals fused. Signal reads
  // also stay after their matching write when they are unused or have a user outside the constrain
  // conditional, because hoisting either read would change the observed component state.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @captured : !felt.type
        struct.member @unused : !felt.type {signal}
        struct.member @external : !felt.type {signal}

        function.def @product(%capture: i1, %unused: i1, %external: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>
          %captured_value = scf.if %capture -> !felt.type {
            %zero = felt.const 0
            scf.yield %zero : !felt.type
          } else {
            %one = felt.const 1
            scf.yield %one : !felt.type
          } {product_source = "compute"}

          %between = arith.constant 0 : index
          scf.if %capture {
            %used = arith.addi %between, %between : index
            scf.yield
          } else {
            %used = arith.addi %between, %between : index
            scf.yield
          } {product_source = "constrain"}
          struct.writem %self[@captured] = %captured_value : <@A>, !felt.type

          %unused_value = scf.if %unused -> !felt.type {
            %zero = felt.const 0
            scf.yield %zero : !felt.type
          } else {
            %one = felt.const 1
            scf.yield %one : !felt.type
          } {product_source = "compute"}

          struct.writem %self[@unused] = %unused_value : <@A>, !felt.type {
            product_source = "compute"
          }
          %unused_read = struct.readm %self[@unused] : <@A>, !felt.type {
            product_source = "constrain"
          }

          scf.if %unused {
            constrain.eq %unused_value, %unused_value : !felt.type, !felt.type
          } else {
            constrain.eq %unused_value, %unused_value : !felt.type, !felt.type
          } {product_source = "constrain"}

          %external_value = scf.if %external -> !felt.type {
            %zero = felt.const 0
            scf.yield %zero : !felt.type
          } else {
            %one = felt.const 1
            scf.yield %one : !felt.type
          } {product_source = "compute"}

          struct.writem %self[@external] = %external_value : <@A>, !felt.type {
            product_source = "compute"
          }
          %external_read = struct.readm %self[@external] : <@A>, !felt.type {
            product_source = "constrain"
          }

          scf.if %external {
            constrain.eq %external_read, %external_value : !felt.type, !felt.type
          } else {
            constrain.eq %external_read, %external_value : !felt.type, !felt.type
          } {product_source = "constrain"}

          constrain.eq %external_read, %external_value : !felt.type, !felt.type
          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  unsigned computeIfs = 0;
  unsigned constrainIfs = 0;
  unsigned fusedIfs = 0;
  module->walk([&](mlir::scf::IfOp ifOp) {
    if (auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source"); source) {
      if (source.getValue() == "compute") {
        ++computeIfs;
      } else if (source.getValue() == "constrain") {
        ++constrainIfs;
      } else if (source.getValue() == "fused") {
        ++fusedIfs;
      }
    }
  });

  EXPECT_EQ(computeIfs, 3U);
  EXPECT_EQ(constrainIfs, 3U);
  EXPECT_EQ(fusedIfs, 0U);
}

TEST_F(FuseProductControlFlowTests, NestedLoopControlMismatchPreventsFusion) {
  // The outer conditionals may fuse, but equal-count loops with different bounds or steps have
  // different induction sequences and must remain separate.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product(%condition: i1) -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %zero = felt.const 0

          scf.if %condition {
            scf.for %i = %c0 to %c2 step %c1 {
              %lower_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "compute"}
            scf.for %i = %c0 to %c2 step %c2 {
              %step_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "compute"}
            scf.yield
          } {product_source = "compute"}

          scf.if %condition {
            scf.for %i = %c1 to %c3 step %c1 {
              %lower_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "constrain"}
            scf.for %i = %c0 to %c1 step %c1 {
              %step_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "constrain"}
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@value] = %zero : <@A>, !felt.type
          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  unsigned fusedIfs = 0;
  unsigned computeLoops = 0;
  unsigned constrainLoops = 0;
  unsigned fusedLoops = 0;
  product.walk([&](mlir::scf::IfOp ifOp) {
    if (auto source = ifOp->getAttrOfType<mlir::StringAttr>("product_source");
        source && source.getValue() == "fused") {
      ++fusedIfs;
    }
  });
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }
    if (source.getValue() == "compute") {
      ++computeLoops;
    } else if (source.getValue() == "constrain") {
      ++constrainLoops;
    } else if (source.getValue() == "fused") {
      ++fusedLoops;
    }
  });

  EXPECT_EQ(fusedIfs, 1U);
  EXPECT_EQ(computeLoops, 2U);
  EXPECT_EQ(constrainLoops, 2U);
  EXPECT_EQ(fusedLoops, 0U);
}

TEST_F(FuseProductControlFlowTests, ReversedLoopPairPreventsFusion) {
  // A constrain loop that precedes its compute partner must remain unchanged; fusion only moves a
  // preceding compute loop toward its constrain partner.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product() -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %zero = felt.const 0

          scf.for %i = %c0 to %c2 step %c1 {
            %observed = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute"}

          struct.writem %self[@value] = %zero : <@A>, !felt.type
          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  mlir::scf::ForOp constrainLoop;
  mlir::scf::ForOp computeLoop;
  unsigned fusedLoops = 0;
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }
    if (source.getValue() == "constrain") {
      constrainLoop = loop;
    } else if (source.getValue() == "compute") {
      computeLoop = loop;
    } else if (source.getValue() == "fused") {
      ++fusedLoops;
    }
  });

  ASSERT_TRUE(constrainLoop);
  ASSERT_TRUE(computeLoop);
  EXPECT_EQ(fusedLoops, 0U);
  EXPECT_TRUE(constrainLoop->isBeforeInBlock(computeLoop));
}

TEST_F(FuseProductControlFlowTests, CrossedLoopPairsUseLexicalApplicationOrder) {
  // The first pair moves the second compute loop past its constrain partner. Rechecking the second
  // pair's current order leaves it unfused, so lexical application order selects A.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product() -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %zero = felt.const 0

          scf.for %i = %c0 to %c2 step %c1 {
            %compute_a = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute"}
          scf.for %i = %c0 to %c3 step %c1 {
            %compute_b = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute"}
          scf.for %i = %c0 to %c3 step %c1 {
            %constrain_b = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}
          scf.for %i = %c0 to %c2 step %c1 {
            %constrain_a = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@value] = %zero : <@A>, !felt.type
          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  mlir::scf::ForOp fusedLoop;
  mlir::scf::ForOp remainingCompute;
  mlir::scf::ForOp remainingConstrain;
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }
    if (source.getValue() == "fused") {
      fusedLoop = loop;
    } else if (source.getValue() == "compute") {
      remainingCompute = loop;
    } else if (source.getValue() == "constrain") {
      remainingConstrain = loop;
    }
  });

  ASSERT_TRUE(fusedLoop);
  ASSERT_TRUE(remainingCompute);
  ASSERT_TRUE(remainingConstrain);
  EXPECT_TRUE(remainingConstrain->isBeforeInBlock(fusedLoop));
  EXPECT_TRUE(fusedLoop->isBeforeInBlock(remainingCompute));
}

TEST_F(FuseProductControlFlowTests, DistinctConstReadsOfSameBindingCanFuse) {
  // Direct fusion must accept distinct same-block reads of one template binding, while a pair
  // naming different bindings remains separate. No earlier pass canonicalizes the two reads.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      poly.template @T {
        poly.param @N : index
        poly.param @M : index
        poly.param @K : index

        struct.def @A {
          struct.member @out : !felt.type

          function.def @product() -> !struct.type<@T::@A<[@N, @M, @K]>> {
            %self = struct.new : <@T::@A<[@N, @M, @K]>>
            %c0 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            %same_compute = poly.read_const @N : index
            %same_constrain = poly.read_const @N : index

            scf.for %i = %c0 to %same_compute step %c1 {
              %same_compute_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "compute"}
            scf.for %i = %c0 to %same_constrain step %c1 {
              %same_constrain_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "constrain"}

            %different_compute = poly.read_const @M : index
            %different_constrain = poly.read_const @K : index
            scf.for %i = %c0 to %different_compute step %c1 {
              %different_compute_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "compute"}
            scf.for %i = %c0 to %different_constrain step %c1 {
              %different_constrain_value = arith.addi %i, %c0 : index
              scf.yield
            } {product_source = "constrain"}

            %zero = felt.const 0
            struct.writem %self[@out] = %zero : <@T::@A<[@N, @M, @K]>>, !felt.type
            function.return %self : !struct.type<@T::@A<[@N, @M, @K]>>
          }
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  mlir::scf::ForOp fusedLoop;
  mlir::scf::ForOp differentComputeLoop;
  mlir::scf::ForOp differentConstrainLoop;
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }
    if (source.getValue() == "fused") {
      fusedLoop = loop;
    } else if (source.getValue() == "compute") {
      differentComputeLoop = loop;
    } else if (source.getValue() == "constrain") {
      differentConstrainLoop = loop;
    }
  });

  ASSERT_TRUE(fusedLoop);
  ASSERT_TRUE(differentComputeLoop);
  ASSERT_TRUE(differentConstrainLoop);

  auto fusedUpperBound = fusedLoop.getUpperBound().getDefiningOp<llzk::polymorphic::ConstReadOp>();
  auto differentComputeUpperBound =
      differentComputeLoop.getUpperBound().getDefiningOp<llzk::polymorphic::ConstReadOp>();
  auto differentConstrainUpperBound =
      differentConstrainLoop.getUpperBound().getDefiningOp<llzk::polymorphic::ConstReadOp>();
  ASSERT_TRUE(fusedUpperBound);
  ASSERT_TRUE(differentComputeUpperBound);
  ASSERT_TRUE(differentConstrainUpperBound);
  EXPECT_EQ(fusedUpperBound.getConstName(), "N");
  EXPECT_EQ(differentComputeUpperBound.getConstName(), "M");
  EXPECT_EQ(differentConstrainUpperBound.getConstName(), "K");
}

TEST_F(FuseProductControlFlowTests, LoopUnsignedComparisonMustMatch) {
  // Loops that select different signed or unsigned bound comparisons stay separate. Unit and true
  // select unsigned comparison; absent and false select signed comparison, and the fused loop
  // retains the accepted input spelling.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product() -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %c4 = arith.constant 4 : index
          %c5 = arith.constant 5 : index
          %zero = felt.const 0

          scf.for %i = %c0 to %c2 step %c1 {
            %mixed_compute = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute", unsignedCmp}
          scf.for %i = %c0 to %c2 step %c1 {
            %mixed_constrain = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}

          scf.for %i = %c0 to %c3 step %c1 {
            %equal_compute = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute", unsignedCmp}
          scf.for %i = %c0 to %c3 step %c1 {
            %equal_constrain = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain", unsignedCmp}

          scf.for %i = %c0 to %c4 step %c1 {
            %false_compute = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute"}
          scf.for %i = %c0 to %c4 step %c1 {
            %false_constrain = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain", unsignedCmp = false}

          scf.for %i = %c0 to %c5 step %c1 {
            %true_compute = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "compute", unsignedCmp = true}
          scf.for %i = %c0 to %c5 step %c1 {
            %true_constrain = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain", unsignedCmp}

          struct.writem %self[@value] = %zero : <@A>, !felt.type
          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  // Unique upper bounds tie each signed/unsigned expectation to its original loop pair.
  llvm::SmallVector<mlir::scf::ForOp> upperBound2Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound3Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound4Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound5Loops;
  unsigned unexpectedLoopBounds = 0;
  product.walk([&](mlir::scf::ForOp loop) {
    auto upperBound = loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!upperBound) {
      ++unexpectedLoopBounds;
      return;
    }
    switch (llvm::cast<mlir::IntegerAttr>(upperBound.getValue()).getInt()) {
    case 2:
      upperBound2Loops.push_back(loop);
      break;
    case 3:
      upperBound3Loops.push_back(loop);
      break;
    case 4:
      upperBound4Loops.push_back(loop);
      break;
    case 5:
      upperBound5Loops.push_back(loop);
      break;
    default:
      ++unexpectedLoopBounds;
      break;
    }
  });

  EXPECT_EQ(unexpectedLoopBounds, 0U);

  ASSERT_EQ(upperBound2Loops.size(), 2U);
  mlir::StringAttr upperBound2ComputeSource =
      upperBound2Loops[0]->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(upperBound2ComputeSource);
  EXPECT_EQ(upperBound2ComputeSource.getValue(), "compute");
  EXPECT_TRUE(llvm::isa<mlir::UnitAttr>(upperBound2Loops[0]->getAttr("unsignedCmp")));
  mlir::StringAttr upperBound2ConstrainSource =
      upperBound2Loops[1]->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(upperBound2ConstrainSource);
  EXPECT_EQ(upperBound2ConstrainSource.getValue(), "constrain");
  EXPECT_FALSE(upperBound2Loops[1]->hasAttr("unsignedCmp"));

  ASSERT_EQ(upperBound3Loops.size(), 1U);
  mlir::StringAttr upperBound3Source =
      upperBound3Loops[0]->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(upperBound3Source);
  EXPECT_EQ(upperBound3Source.getValue(), "fused");
  EXPECT_TRUE(llvm::isa<mlir::UnitAttr>(upperBound3Loops[0]->getAttr("unsignedCmp")));

  ASSERT_EQ(upperBound4Loops.size(), 1U);
  mlir::StringAttr upperBound4Source =
      upperBound4Loops[0]->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(upperBound4Source);
  EXPECT_EQ(upperBound4Source.getValue(), "fused");
  mlir::BoolAttr falseMode = upperBound4Loops[0]->getAttrOfType<mlir::BoolAttr>("unsignedCmp");
  ASSERT_TRUE(falseMode);
  EXPECT_FALSE(falseMode.getValue());

  ASSERT_EQ(upperBound5Loops.size(), 1U);
  mlir::StringAttr upperBound5Source =
      upperBound5Loops[0]->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(upperBound5Source);
  EXPECT_EQ(upperBound5Source.getValue(), "fused");
  mlir::BoolAttr trueMode = upperBound5Loops[0]->getAttrOfType<mlir::BoolAttr>("unsignedCmp");
  ASSERT_TRUE(trueMode);
  EXPECT_TRUE(trueMode.getValue());
}

TEST_F(FuseProductControlFlowTests, ComputeLoopResultDependenciesPreventFusion) {
  // A compute-loop result cannot move across a surviving constrain-loop user. The final pair is
  // the opposite branch: its pure user has the compute role, moves with the result, and remains
  // fusible.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product() -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c3 = arith.constant 3 : index
          %c4 = arith.constant 4 : index
          %c5 = arith.constant 5 : index
          %c6 = arith.constant 6 : index
          %c7 = arith.constant 7 : index
          %zero = felt.const 0

          // The constrain iter_arg is initialized from the compute-loop result.
          %compute_iter = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %c0) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "compute"}
          scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %compute_iter) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "constrain"}

          // A nested constrain bound captures the compute-loop result.
          %compute_bound = scf.for %i = %c0 to %c3 step %c1 iter_args(%acc = %c0) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "compute"}
          scf.for %i = %c0 to %c3 step %c1 {
            scf.for %j = %c0 to %compute_bound step %c1 {
              %nested = arith.addi %j, %c1 : index
              scf.yield
            }
            scf.yield
          } {product_source = "constrain"}

          // A pure operation in the constrain body captures the compute-loop result.
          %compute_body = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "compute"}
          scf.for %i = %c0 to %c4 step %c1 {
            %captured = arith.addi %compute_body, %c1 : index
            scf.yield
          } {product_source = "constrain"}

          // A pure intervening operation with the constrain role uses the result before the loop.
          %compute_tagged = scf.for %i = %c0 to %c5 step %c1 iter_args(%acc = %c0) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "compute"}
          %constrain_use = arith.addi %compute_tagged, %c1 {product_source = "constrain"} : index
          scf.for %i = %c0 to %c5 step %c1 {
            %unused_tagged = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}

          // An unmarked pure operation between the loops has the same surviving-use restriction.
          %compute_unmarked = scf.for %i = %c0 to %c6 step %c1 iter_args(%acc = %c0) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "compute"}
          %unmarked_use = arith.addi %compute_unmarked, %c1 : index
          scf.for %i = %c0 to %c6 step %c1 {
            %unused_unmarked = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}

          // The pure use with the compute role moves after the constrain loop with its definition.
          %compute_sink = scf.for %i = %c0 to %c7 step %c1 iter_args(%acc = %c0) -> (index) {
            %next = arith.addi %acc, %c1 : index
            scf.yield %next : index
          } {product_source = "compute"}
          %sink_use = arith.addi %compute_sink, %c1 {product_source = "compute"} : index
          scf.for %i = %c0 to %c7 step %c1 {
            %unused_sink = arith.addi %i, %c0 : index
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@value] = %zero : <@A>, !felt.type
          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  // Unique upper bounds identify every original pair, including the one legal fusion.
  llvm::SmallVector<mlir::scf::ForOp> upperBound2Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound3Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound4Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound5Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound6Loops;
  llvm::SmallVector<mlir::scf::ForOp> upperBound7Loops;
  unsigned unexpectedMarkedLoopBounds = 0;
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }

    auto upperBound = loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!upperBound) {
      ++unexpectedMarkedLoopBounds;
      return;
    }
    switch (llvm::cast<mlir::IntegerAttr>(upperBound.getValue()).getInt()) {
    case 2:
      upperBound2Loops.push_back(loop);
      break;
    case 3:
      upperBound3Loops.push_back(loop);
      break;
    case 4:
      upperBound4Loops.push_back(loop);
      break;
    case 5:
      upperBound5Loops.push_back(loop);
      break;
    case 6:
      upperBound6Loops.push_back(loop);
      break;
    case 7:
      upperBound7Loops.push_back(loop);
      break;
    default:
      ++unexpectedMarkedLoopBounds;
      break;
    }
  });

  EXPECT_EQ(unexpectedMarkedLoopBounds, 0U);

  for (auto *loopPair : {
           &upperBound2Loops,
           &upperBound3Loops,
           &upperBound4Loops,
           &upperBound5Loops,
           &upperBound6Loops,
       }) {
    ASSERT_EQ(loopPair->size(), 2U);
    mlir::StringAttr computeSource =
        (*loopPair)[0]->getAttrOfType<mlir::StringAttr>("product_source");
    mlir::StringAttr constrainSource =
        (*loopPair)[1]->getAttrOfType<mlir::StringAttr>("product_source");
    ASSERT_TRUE(computeSource);
    ASSERT_TRUE(constrainSource);
    EXPECT_EQ(computeSource.getValue(), "compute");
    EXPECT_EQ(constrainSource.getValue(), "constrain");
    EXPECT_TRUE((*loopPair)[0]->isBeforeInBlock((*loopPair)[1]));
  }

  EXPECT_EQ(upperBound2Loops[0].getNumResults(), 1U);
  EXPECT_EQ(upperBound2Loops[1].getNumResults(), 1U);
  ASSERT_EQ(upperBound2Loops[1].getInitArgs().size(), 1U);
  EXPECT_TRUE(upperBound2Loops[1].getInitArgs().front() == upperBound2Loops[0].getResult(0));
  for (auto *loopPair : {
           &upperBound3Loops,
           &upperBound4Loops,
           &upperBound5Loops,
           &upperBound6Loops,
       }) {
    EXPECT_EQ((*loopPair)[0].getNumResults(), 1U);
    EXPECT_EQ((*loopPair)[1].getNumResults(), 0U);
  }

  mlir::arith::AddIOp taggedUse;
  mlir::arith::AddIOp unmarkedUse;
  product.walk([&](mlir::arith::AddIOp add) {
    if (add.getLhs() == upperBound5Loops[0].getResult(0)) {
      taggedUse = add;
    } else if (add.getLhs() == upperBound6Loops[0].getResult(0)) {
      unmarkedUse = add;
    }
  });
  ASSERT_TRUE(taggedUse);
  ASSERT_TRUE(unmarkedUse);
  mlir::StringAttr taggedSource = taggedUse->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(taggedSource);
  EXPECT_EQ(taggedSource.getValue(), "constrain");
  EXPECT_FALSE(unmarkedUse->hasAttr("product_source"));
  EXPECT_TRUE(upperBound5Loops[0]->isBeforeInBlock(taggedUse));
  EXPECT_TRUE(taggedUse->isBeforeInBlock(upperBound5Loops[1]));
  EXPECT_TRUE(upperBound6Loops[0]->isBeforeInBlock(unmarkedUse));
  EXPECT_TRUE(unmarkedUse->isBeforeInBlock(upperBound6Loops[1]));

  ASSERT_EQ(upperBound7Loops.size(), 1U);
  mlir::scf::ForOp fusedLoop = upperBound7Loops.front();
  mlir::StringAttr fusedSource = fusedLoop->getAttrOfType<mlir::StringAttr>("product_source");
  ASSERT_TRUE(fusedSource);
  EXPECT_EQ(fusedSource.getValue(), "fused");
  ASSERT_EQ(fusedLoop.getRegionIterArgs().size(), 1U);
  ASSERT_EQ(fusedLoop.getNumResults(), 1U);

  llvm::SmallVector<mlir::arith::AddIOp> fusedBodyAdds;
  for (mlir::Operation &op : *fusedLoop.getBody()) {
    if (auto add = llvm::dyn_cast<mlir::arith::AddIOp>(&op)) {
      fusedBodyAdds.push_back(add);
    }
  }
  ASSERT_EQ(fusedBodyAdds.size(), 2U);
  EXPECT_TRUE(fusedBodyAdds[0].getLhs() == fusedLoop.getRegionIterArgs().front());
  EXPECT_TRUE(fusedBodyAdds[0].getRhs() == fusedLoop.getStep());
  EXPECT_TRUE(fusedBodyAdds[1].getLhs() == fusedLoop.getInductionVar());
  EXPECT_TRUE(fusedBodyAdds[1].getRhs() == fusedLoop.getLowerBound());
  EXPECT_TRUE(fusedBodyAdds[0]->isBeforeInBlock(fusedBodyAdds[1]));

  llvm::SmallVector<mlir::arith::AddIOp> sunkComputeUses;
  product.walk([&](mlir::arith::AddIOp add) {
    mlir::StringAttr source = add->getAttrOfType<mlir::StringAttr>("product_source");
    if (source && source.getValue() == "compute") {
      sunkComputeUses.push_back(add);
    }
  });
  ASSERT_EQ(sunkComputeUses.size(), 1U);
  EXPECT_TRUE(sunkComputeUses.front().getLhs() == fusedLoop.getResult(0));
  EXPECT_TRUE(fusedLoop->isBeforeInBlock(sunkComputeUses.front()));
}

TEST_F(FuseProductControlFlowTests, EffectfulOperationsBetweenLoopsPreventFusion) {
  // Effectful global and RAM operations between otherwise-admissible loops must remain in their
  // original order, whether the intervening effect has the compute role, the constrain role, or
  // no role. Distinct parent structs and typed barriers identify each original loop pair
  // independently.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      global.def @g : !felt.type = 0

      struct.def @GlobalCase {
        struct.member @out : !felt.type

        function.def @product(%value: !felt.type) -> !struct.type<@GlobalCase> {
          %self = struct.new : <@GlobalCase>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %one = felt.const 1

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %value, %one
            scf.yield
          } {product_source = "compute"}

          %stored = felt.const 2 {product_source = "compute"}
          global.write @g = %stored : !felt.type {product_source = "compute"}

          scf.for %i = %c0 to %c2 step %c1 {
            constrain.eq %value, %one : !felt.type
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@out] = %value : <@GlobalCase>, !felt.type
          function.return %self : !struct.type<@GlobalCase>
        }
      }

      struct.def @RamCase {
        struct.member @out : !felt.type

        function.def @product(%value: !felt.type) -> !struct.type<@RamCase> {
          %self = struct.new : <@RamCase>
          %addr = arith.constant 0 : index
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %one = felt.const 1

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %value, %one
            scf.yield
          } {product_source = "compute"}

          %stored = felt.const 2 {product_source = "compute"}
          ram.store %addr, %stored : !felt.type {product_source = "compute"}

          scf.for %i = %c0 to %c2 step %c1 {
            constrain.eq %value, %one : !felt.type
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@out] = %value : <@RamCase>, !felt.type
          function.return %self : !struct.type<@RamCase>
        }
      }

      struct.def @ConstrainEffectCase {
        struct.member @out : !felt.type

        function.def @product(%value: !felt.type) -> !struct.type<@ConstrainEffectCase> {
          %self = struct.new : <@ConstrainEffectCase>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %one = felt.const 1

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %value, %one
            scf.yield
          } {product_source = "compute"}

          %stored = felt.const 3
          global.write @g = %stored : !felt.type {product_source = "constrain"}

          scf.for %i = %c0 to %c2 step %c1 {
            constrain.eq %value, %one : !felt.type
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@out] = %value : <@ConstrainEffectCase>, !felt.type
          function.return %self : !struct.type<@ConstrainEffectCase>
        }
      }

      struct.def @UnmarkedEffectCase {
        struct.member @out : !felt.type

        function.def @product(%value: !felt.type) -> !struct.type<@UnmarkedEffectCase> {
          %self = struct.new : <@UnmarkedEffectCase>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %one = felt.const 1

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %value, %one
            scf.yield
          } {product_source = "compute"}

          %stored = felt.const 4
          global.write @g = %stored : !felt.type

          scf.for %i = %c0 to %c2 step %c1 {
            constrain.eq %value, %one : !felt.type
            scf.yield
          } {product_source = "constrain"}

          struct.writem %self[@out] = %value : <@UnmarkedEffectCase>, !felt.type
          function.return %self : !struct.type<@UnmarkedEffectCase>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llvm::SmallVector<llzk::function::FuncDefOp> products;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      products.push_back(func);
    }
  });
  ASSERT_EQ(products.size(), 4U);

  for (llzk::function::FuncDefOp product : products) {
    llzk::component::StructDefOp parent = product->getParentOfType<llzk::component::StructDefOp>();
    ASSERT_TRUE(parent);

    llvm::SmallVector<mlir::scf::ForOp> loops;
    product.walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr("product_source")) {
        loops.push_back(loop);
      }
    });
    ASSERT_EQ(loops.size(), 2U);
    mlir::StringAttr computeSource = loops[0]->getAttrOfType<mlir::StringAttr>("product_source");
    mlir::StringAttr constrainSource = loops[1]->getAttrOfType<mlir::StringAttr>("product_source");
    ASSERT_TRUE(computeSource);
    ASSERT_TRUE(constrainSource);
    EXPECT_EQ(computeSource.getValue(), "compute");
    EXPECT_EQ(constrainSource.getValue(), "constrain");
    EXPECT_TRUE(loops[0]->isBeforeInBlock(loops[1]));

    llvm::SmallVector<mlir::Operation *> barriers;
    product.walk([&](llzk::global::GlobalWriteOp write) {
      barriers.push_back(write.getOperation());
    });
    product.walk([&](llzk::ram::StoreOp store) { barriers.push_back(store.getOperation()); });
    ASSERT_EQ(barriers.size(), 1U);
    mlir::Operation *barrier = barriers.front();
    EXPECT_EQ(barrier->getBlock(), loops[0]->getBlock());
    EXPECT_TRUE(loops[0]->isBeforeInBlock(barrier));
    EXPECT_TRUE(barrier->isBeforeInBlock(loops[1]));

    mlir::StringRef caseName = parent.getName();
    mlir::StringAttr barrierSource = barrier->getAttrOfType<mlir::StringAttr>("product_source");
    bool isRamCase = caseName == "RamCase";
    EXPECT_EQ(llvm::isa<llzk::ram::StoreOp>(barrier), isRamCase);
    EXPECT_EQ(llvm::isa<llzk::global::GlobalWriteOp>(barrier), !isRamCase);

    mlir::StringRef expectedSource;
    if (caseName == "GlobalCase" || isRamCase) {
      expectedSource = "compute";
    } else if (caseName == "ConstrainEffectCase") {
      expectedSource = "constrain";
    } else if (caseName != "UnmarkedEffectCase") {
      ADD_FAILURE() << "unexpected effect case: " << caseName.str();
    }

    if (expectedSource.empty()) {
      EXPECT_FALSE(barrierSource);
    } else {
      ASSERT_TRUE(barrierSource);
      EXPECT_EQ(barrierSource.getValue(), expectedSource);
    }
  }
}

TEST_F(FuseProductControlFlowTests, InterveningMemberReadPreventsMemberWriteSink) {
  // A non-signal read between the loops is the only rejection reason: sinking the preceding write
  // would make the read observe the old member value instead of the value written before it.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product(%input: !felt.type) -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %input, %input
            scf.yield
          } {product_source = "compute"}

          struct.writem %self[@value] = %input : <@A>, !felt.type {
            product_source = "compute"
          }
          %observed = struct.readm %self[@value] : <@A>, !felt.type

          scf.for %i = %c0 to %c2 step %c1 {
            constrain.eq %observed, %input : !felt.type
            scf.yield
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  mlir::scf::ForOp computeLoop;
  mlir::scf::ForOp constrainLoop;
  mlir::scf::ForOp fusedLoop;
  llzk::component::MemberWriteOp write;
  llzk::component::MemberReadOp read;
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }
    if (source.getValue() == "compute") {
      computeLoop = loop;
    } else if (source.getValue() == "constrain") {
      constrainLoop = loop;
    } else if (source.getValue() == "fused") {
      fusedLoop = loop;
    }
  });
  product.walk([&](llzk::component::MemberWriteOp writeOp) { write = writeOp; });
  product.walk([&](llzk::component::MemberReadOp readOp) { read = readOp; });

  ASSERT_TRUE(computeLoop);
  ASSERT_TRUE(constrainLoop);
  ASSERT_TRUE(write);
  ASSERT_TRUE(read);
  EXPECT_FALSE(fusedLoop);
  EXPECT_TRUE(computeLoop->isBeforeInBlock(write));
  EXPECT_TRUE(write->isBeforeInBlock(read));
  EXPECT_TRUE(read->isBeforeInBlock(constrainLoop));
}

TEST_F(FuseProductControlFlowTests, MemberWriteSinkDoesNotCrossConstrainRead) {
  // A member write may be sunk only when the constrain loop cannot observe component state. This
  // body-effect interaction case is distinct from the intervening-read interval case above.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product() -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %value = felt.const 7

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %value, %value
            scf.yield
          } {product_source = "compute"}

          struct.writem %self[@value] = %value : <@A>, !felt.type {
            product_source = "compute"
          }

          scf.for %i = %c0 to %c2 step %c1 {
            %observed = struct.readm %self[@value] : <@A>, !felt.type {
              product_source = "constrain"
            }
            constrain.eq %observed, %value : !felt.type, !felt.type
            scf.yield
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  llzk::component::MemberWriteOp write;
  llzk::component::MemberReadOp read;
  mlir::scf::ForOp constrainLoop;
  unsigned fusedLoops = 0;
  product.walk([&](llzk::component::MemberWriteOp writeOp) { write = writeOp; });
  product.walk([&](llzk::component::MemberReadOp readOp) { read = readOp; });
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (!source) {
      return;
    }
    if (source.getValue() == "constrain") {
      constrainLoop = loop;
    } else if (source.getValue() == "fused") {
      ++fusedLoops;
    }
  });

  ASSERT_TRUE(write);
  ASSERT_TRUE(read);
  ASSERT_TRUE(constrainLoop);
  EXPECT_EQ(fusedLoops, 0U);
  EXPECT_TRUE(write->isBeforeInBlock(constrainLoop));
}

TEST_F(FuseProductControlFlowTests, MemberWriteSinkRemainsFusibleWithoutConstrainRead) {
  // Direct member writes retain their existing sink behavior when the constrain loop has no
  // storage or unknown effect that could observe the write.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::parseSourceString<mlir::ModuleOp>(
      R"mlir(
    module attributes {llzk.lang = "llzk"} {
      struct.def @A {
        struct.member @value : !felt.type

        function.def @product(%input: !felt.type) -> !struct.type<@A> {
          %self = struct.new : <@A>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %c2 = arith.constant 2 : index

          scf.for %i = %c0 to %c2 step %c1 {
            %computed = felt.add %input, %input
            scf.yield
          } {product_source = "compute"}

          struct.writem %self[@value] = %input : <@A>, !felt.type {
            product_source = "compute"
          }

          scf.for %i = %c0 to %c2 step %c1 {
            %expected = felt.const 0
            constrain.eq %input, %expected : !felt.type, !felt.type
            scf.yield
          } {product_source = "constrain"}

          function.return %self : !struct.type<@A>
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  mlir::PassManager pm(&ctx);
  pm.addPass(llzk::createFuseProductControlFlowPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  llzk::function::FuncDefOp product;
  module->walk([&](llzk::function::FuncDefOp func) {
    if (func.isStructProduct()) {
      product = func;
    }
  });
  ASSERT_TRUE(product);

  llzk::component::MemberWriteOp write;
  mlir::scf::ForOp fusedLoop;
  product.walk([&](llzk::component::MemberWriteOp writeOp) { write = writeOp; });
  product.walk([&](mlir::scf::ForOp loop) {
    mlir::StringAttr source = loop->getAttrOfType<mlir::StringAttr>("product_source");
    if (source && source.getValue() == "fused") {
      fusedLoop = loop;
    }
  });

  ASSERT_TRUE(write);
  ASSERT_TRUE(fusedLoop);
  EXPECT_TRUE(fusedLoop->isBeforeInBlock(write));
}

} // namespace
