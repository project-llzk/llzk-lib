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

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_include) {
  (void)mlirGetDialectHandle__llzk__include__();
}

TEST_F(CAPITest, llzk_include_op_create) {
  auto location = mlirLocationUnknownGet(context);
  auto op = llzkInclude_IncludeOpCreate(
      location, mlirStringRefCreateFromCString("test"), mlirStringRefCreateFromCString("test.mlir")
  );

  EXPECT_NE(op.ptr, (void *)NULL);
  mlirOperationDestroy(op);
}
