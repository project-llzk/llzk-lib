//===-- Include.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Include.h"

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Include/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Include/IR/Ops.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_include_op_create) {
  auto location = mlirLocationUnknownGet(context);
  auto op = llzkInclude_IncludeOpCreateInferredContext(
      location, mlirStringRefCreateFromCString("test"), mlirStringRefCreateFromCString("test.mlir")
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  mlirOperationDestroy(op);
}

// Implementation for `IncludeOp_build_pass` test
std::unique_ptr<IncludeOpBuildFuncHelper> IncludeOpBuildFuncHelper::get() {
  struct Impl : public IncludeOpBuildFuncHelper {
    MlirModule parentModule;
    ~Impl() override { mlirModuleDestroy(this->parentModule); }
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Need to create a parent module because `include.from` requires it for verification
      this->parentModule = mlirModuleCreateEmpty(location);
      mlirOpBuilderSetInsertionPointToStart(builder, mlirModuleGetBody(this->parentModule));
      MlirIdentifier strVal =
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString("test"));
      return llzkInclude_IncludeOpBuild(builder, location, strVal, strVal);
    }
  };
  return std::make_unique<Impl>();
}
