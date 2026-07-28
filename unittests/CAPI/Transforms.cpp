//===-- Transforms.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Transforms.h"

#include "CAPITestBase.h"

#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"

using namespace mlir;
using namespace llzk;

TEST_F(CAPITest, RegisterTransformationPassesAndCreate) {
  mlirRegisterLLZKTransformationPasses();
  auto manager = mlirPassManagerCreate(context);

  auto pass1 = mlirCreateLLZKTransformationRedundantOperationEliminationPass();
  auto pass2 = mlirCreateLLZKTransformationRedundantReadAndWriteEliminationPass();
  auto pass3 = mlirCreateLLZKTransformationUnusedDeclarationEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass1);
  mlirPassManagerAddOwnedPass(manager, pass2);
  mlirPassManagerAddOwnedPass(manager, pass3);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, RegisterRedundantOperationEliminationPassAndCreate) {
  mlirRegisterLLZKTransformationRedundantOperationEliminationPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKTransformationRedundantOperationEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, RegisterRedudantReadAndWriteEliminationPassAndCreate) {
  mlirRegisterLLZKTransformationRedundantReadAndWriteEliminationPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKTransformationRedundantReadAndWriteEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}

TEST_F(CAPITest, RegisterUnuusedDeclarationEliminationPassAndCreate) {
  mlirRegisterLLZKTransformationUnusedDeclarationEliminationPass();
  auto manager = mlirPassManagerCreate(context);

  auto pass = mlirCreateLLZKTransformationUnusedDeclarationEliminationPass();
  mlirPassManagerAddOwnedPass(manager, pass);

  mlirPassManagerDestroy(manager);
}

static polymorphic::TemplateExprOp buildPolyExpr(MLIRContext *ctx, bool includeDeadValue = false) {
  auto loc = UnknownLoc::get(ctx);

  OpBuilder bldr(ctx);
  auto expr = bldr.create<polymorphic::TemplateExprOp>(loc, bldr.getStringAttr("Sub_12@269"));
  bldr.setInsertionPointToStart(&expr.getInitializerRegion().emplaceBlock());

  auto feltTy = felt::FeltType::get(ctx, bldr.getStringAttr("bn128"));
  auto const12 = felt::FeltConstAttr::get(ctx, APInt(64, 12), feltTy);
  if (includeDeadValue) {
    // Create a dead value that should be eliminated by the DVE pass.
    // Suppress `clang-tidy` since it's intentional.
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto _ = bldr.create<felt::FeltConstantOp>(loc, const12);
  }
  auto constOp = bldr.create<felt::FeltConstantOp>(loc, const12);
  auto negOp = bldr.create<felt::NegFeltOp>(loc, constOp.getResult());
  bldr.create<polymorphic::YieldOp>(loc, negOp.getResult());

  return expr;
}

static std::string toString(auto op) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  op.print(os, OpPrintingFlags().assumeVerified().printGenericOpForm(false));
  return buffer;
}

static void printToStderr(MlirStringRef str, void *userData) {
  (void)userData;
  fwrite(str.data, 1, str.length, stderr);
}

/// Setup pass manager and run dead value elimination. Replicate how it's done in the circom
/// frontend via `llzk-rs`.
///
/// Returns true on success.
static bool runRemoveDeadValues(MlirContext context, MlirOperation op) {
  MlirPassManager pm = mlirPassManagerCreate(context); // PassManager::new()
  mlirRegisterAllPasses();                             // utility::register_all_passes()
  mlirPassManagerEnableVerifier(pm, false);            // enable_verifier(false)
  auto opm = mlirPassManagerGetAsOpPassManager(pm);    // as_operation_pass_manager()
  auto pipeline = mlirStringRefCreateFromCString("poly.expr(remove-dead-values)");
  mlirParsePassPipeline(opm, pipeline, printToStderr, NULL); // utility::parse_pass_pipeline()
  bool success = mlirPassManagerRunOnOp(pm, op).value != 0;
  mlirPassManagerDestroy(pm);
  return success;
}

TEST_F(CAPITest, PolyExprDeadValueEliminationNoChange) {
  auto exprOp = buildPolyExpr(unwrap(context));
  std::string before = toString(exprOp);

  ASSERT_TRUE(runRemoveDeadValues(context, wrap(exprOp))) << "failed to run pass pipeline";

  // There were no dead values to eliminate, so the IR should be unchanged.
  ASSERT_EQ(before, toString(exprOp));

  // Cleanup
  mlirOperationDestroy(wrap(exprOp));
}

TEST_F(CAPITest, PolyExprDeadValueEliminationWithChange) {
  auto exprOp = buildPolyExpr(unwrap(context), true);
  std::string before = toString(exprOp);

  // Setup pass manager and run dead value elimination.
  // Replicate how it's done in the circom frontend via `llzk-rs`
  ASSERT_TRUE(runRemoveDeadValues(context, wrap(exprOp))) << "failed to run pass pipeline";

  // The dead value should have been eliminated.
  std::string after = toString(exprOp);
  ASSERT_NE(before, after);
  // It's the same as if built without the dead value in the first place.
  ASSERT_EQ(after, toString(buildPolyExpr(unwrap(context))));

  // Cleanup
  mlirOperationDestroy(wrap(exprOp));
}
