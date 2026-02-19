//===-- Undef.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Undef.h"

#include "llzk-c/Dialect/Felt.h"

#include <mlir-c/IR.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Undef/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Undef/IR/Ops.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_operation_is_a_undef_op_pass) {
  auto op_name = mlirStringRefCreateFromCString("undef.undef");
  auto state = mlirOperationStateGet(op_name, mlirLocationUnknownGet(context));
  auto t = llzkFelt_FeltTypeGet(context);
  mlirOperationStateAddResults(&state, 1, &t);

  auto op = mlirOperationCreate(&state);
  EXPECT_TRUE(llzkOperationIsA_Undef_UndefOp(op));
}

// Implementation for `UndefOp_build_pass` test
std::unique_ptr<UndefOpBuildFuncHelper> UndefOpBuildFuncHelper::get() {
  struct Impl : public UndefOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      return llzkUndef_UndefOpBuild(builder, location, llzkFelt_FeltTypeGet(testClass.context));
    }
  };
  return std::make_unique<Impl>();
}
