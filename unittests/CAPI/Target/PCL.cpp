//===-- PCL.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Target/PCL.h"

#include "../CAPITestBase.h"
#include "pcl/Dialect/IR/Attrs.h"
#include "pcl/Dialect/IR/Ops.h"
#include "pcl/Dialect/IR/Types.h"
#include "pcl/DialectRegistration.h"

#include "llzk/Config/Config.h"

#include <mlir-c/IR.h>

#include <mlir/CAPI/Support.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/Debug.h>

constexpr unsigned SEVEN = 7;

TEST_F(CAPITest, FullPCLLoweringPipelineCAPI) {
  mlir::DialectRegistry registry;
  pcl::registerDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  mlirContextLoadAllAvailableDialects(context);
  llzkRegisterPCLConversionPassPipelines();

  constexpr const char *source = R"mlir(
    module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {
      struct.def @Main {
        struct.member @out : !array.type<2 x !felt.type<"bn128">> {llzk.pub}
        function.def @compute(%in: !felt.type<"bn128">) -> !struct.type<@Main> {
          %self = struct.new : <@Main>
          function.return %self : !struct.type<@Main>
        }
        function.def @constrain(%self: !struct.type<@Main>, %in: !felt.type<"bn128">) {
          %out = struct.readm %self[@out] : <@Main>, !array.type<2 x !felt.type<"bn128">>
          %c0 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          %a = array.read %out[%c0] : <2 x !felt.type<"bn128">>, !felt.type<"bn128">
          %b = array.read %out[%c1] : <2 x !felt.type<"bn128">>, !felt.type<"bn128">
          constrain.eq %a, %in : !felt.type<"bn128">, !felt.type<"bn128">
          constrain.eq %b, %in : !felt.type<"bn128">, !felt.type<"bn128">
          function.return
        }
      }
    }
  )mlir";

  // Exercise both direct construction and registration for textual pipelines.
  for (bool parsePipeline : {false, true}) {
    auto module = mlirModuleCreateParse(context, mlirStringRefCreateFromCString(source));
    ASSERT_FALSE(mlirModuleIsNull(module));
    auto manager = mlirPassManagerCreate(context);
    auto opManager = mlirPassManagerGetAsOpPassManager(manager);
    if (parsePipeline) {
      std::string diagnostics;
      auto callback = [](MlirStringRef text, void *data) {
        static_cast<std::string *>(data)->append(text.data, text.length);
      };
      EXPECT_TRUE(mlirLogicalResultIsSuccess(mlirParsePassPipeline(
          opManager, mlirStringRefCreateFromCString("builtin.module(llzk-full-pcl-lowering)"),
          callback, &diagnostics
      ))) << diagnostics;
    } else {
      llzkBuildFullPCLLoweringPipeline(opManager);
    }
    EXPECT_TRUE(
        mlirLogicalResultIsSuccess(mlirPassManagerRunOnOp(manager, mlirModuleGetOperation(module)))
    );
    auto lowered = unwrap(module);
    EXPECT_TRUE(lowered->hasAttr(PCL_PRIME_ATTR_NAME));
    auto functions = lowered.getOps<mlir::func::FuncOp>();
    ASSERT_FALSE(functions.empty());
    EXPECT_EQ((*functions.begin()).getFunctionType().getNumResults(), 2u);
    unsigned assertions = 0;
    lowered.walk([&](pcl::AssertOp) { ++assertions; });
    EXPECT_EQ(assertions, 2u);
    mlirPassManagerDestroy(manager);
    mlirModuleDestroy(module);
  }
}

constexpr std::string_view EXPECTED_PCL_MODULE_1 =
    // clang-format off
    "(prime-number 7)\n\n"
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
  pcl::registerDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  mlir::OpBuilder builder(ctx);

  auto module = mlir::ModuleOp::create(builder, builder.getUnknownLoc());
  module->setDiscardableAttr(
      builder.getStringAttr(PCL_PRIME_ATTR_NAME),
      pcl::PrimeAttr::get(ctx, llvm::DynamicAPInt(SEVEN))
  );
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());

    auto func = mlir::func::FuncOp::create(
        builder, builder.getUnknownLoc(), builder.getStringAttr("A"),
        builder.getFunctionType({pcl::FeltType::get(ctx)}, {pcl::FeltType::get(ctx)})
    );

    auto &region = func.getBody();
    auto &block = region.emplaceBlock();
    block.addArgument(pcl::FeltType::get(ctx), builder.getUnknownLoc());
    {
      mlir::OpBuilder::InsertionGuard funcGuard(builder);
      builder.setInsertionPointToStart(&block);

      auto inVar = block.getArgument(0);
      auto outVar = pcl::VarOp::create(builder, builder.getUnknownLoc(), "out", true);
      auto eq = pcl::CmpEqOp::create(builder, builder.getUnknownLoc(), inVar, outVar);
      pcl::AssertOp::create(builder, builder.getUnknownLoc(), eq);

      mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(), mlir::ValueRange({outVar}));
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
