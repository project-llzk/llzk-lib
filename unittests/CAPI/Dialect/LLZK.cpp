//===-- LLZK.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/LLZK.h"

#include "llzk-c/Dialect/Felt.h"

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/LLZK/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/LLZK/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/LLZK/IR/Ops.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_public_attr_get) {
  auto attr = llzkLlzk_PublicAttrGet(context);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_attribute_is_a_public_attr_pass) {
  auto attr = llzkLlzk_PublicAttrGet(context);
  EXPECT_TRUE(llzkAttributeIsA_Llzk_PublicAttr(attr));
}

TEST_F(CAPITest, llzk_operation_is_a_nondet_op_pass) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);

  MlirOperation op = llzkLlzk_NonDetOpBuild(builder, location, createIndexType());

  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(llzkOperationIsA_Llzk_NonDetOp(op));

  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}
