//===-- PCL.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Config/Config.h"

#include "llzk-c/Target/PCL.h"

#include <pcl/Dialect/IR/Attrs.h>
#include <pcl/Dialect/IR/Ops.h>
#include <pcl/Dialect/IR/Types.h>
#include <pcl/InitAllDialects.h>

#include <mlir/CAPI/Support.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <mlir-c/IR.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Debug.h>

#include "../CAPITestBase.h"

constexpr unsigned SEVEN = 7;

constexpr std::string_view EXPECTED_PCL_MODULE_1 =
    // clang-format off
    "(prime-number 7)\n"
    "(begin-module A)\n"
    "(input in0)\n"
    "(assert (= in0 out))\n"
    "(output out)\n"
    "(end-module)\n\n"
    // clang-format on
    ;

TEST_F(CAPITest, exportPclModule) {
  std::string output_text;
  llvm::raw_string_ostream ss(output_text);
  auto *ctx = unwrap(context);
  mlir::DialectRegistry registry;
  pcl::registerAllDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  mlir::OpBuilder builder(ctx);

  auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
  module->setDiscardableAttr(
      builder.getStringAttr("pcl.prime"),
      pcl::PrimeAttr::get(ctx, builder.getIntegerAttr(builder.getIntegerType(4), SEVEN))
  );
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), builder.getStringAttr("A"),
        builder.getFunctionType({pcl::FeltType::get(ctx)}, {pcl::FeltType::get(ctx)})
    );

    auto &region = func.getBody();
    auto &block = region.emplaceBlock();
    block.addArgument(pcl::FeltType::get(ctx), builder.getUnknownLoc());
    {
      mlir::OpBuilder::InsertionGuard funcGuard(builder);
      builder.setInsertionPointToStart(&block);

      auto inVar = block.getArgument(0);
      auto outVar = builder.create<pcl::VarOp>(builder.getUnknownLoc(), "out", true);
      auto eq = builder.create<pcl::CmpEqOp>(builder.getUnknownLoc(), inVar, outVar);
      builder.create<pcl::AssertOp>(builder.getUnknownLoc(), eq);

      builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(), mlir::ValueRange({outVar}));
    }
  }


  auto wrappedModule = wrap(module);

  auto result = unwrap(llzkTranslateModuleToPCL(
      mlirModuleGetOperation(wrappedModule), [](MlirStringRef chunk, void *userDataPtr) {
    auto &outstream = *reinterpret_cast<llvm::raw_string_ostream *>(userDataPtr);
    auto text = unwrap(chunk);
    outstream << text;
  }, reinterpret_cast<void *>(&ss)
  ));

#if LLZK_WITH_PCL
  EXPECT_TRUE(mlir::succeeded(result));

  EXPECT_EQ(EXPECTED_PCL_MODULE_1, output_text);
#else
  EXPECT_TRUE(mlir::failed(result));
#endif
}
