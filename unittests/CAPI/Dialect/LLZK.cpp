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

TEST_F(CAPITest, mlir_get_dialect_handle_llzk) { (void)mlirGetDialectHandle__llzk__(); }

TEST_F(CAPITest, llzk_public_attr_get) {
  auto attr = llzkLlzk_PublicAttrGet(context);
  EXPECT_NE(attr.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_attribute_is_a_public_attr_pass) {
  auto attr = llzkLlzk_PublicAttrGet(context);
  EXPECT_TRUE(llzkAttributeIsA_Llzk_PublicAttr(attr));
}

TEST_F(CAPITest, llzk_operation_is_a_nondet_op_pass) {
  auto op_name = mlirStringRefCreateFromCString("llzk.nondet");
  auto state = mlirOperationStateGet(op_name, mlirLocationUnknownGet(context));
  auto t = llzkFelt_FeltTypeGet(context);
  mlirOperationStateAddResults(&state, 1, &t);

  auto op = mlirOperationCreate(&state);
  EXPECT_TRUE(llzkOperationIsA_Llzk_NonDetOp(op));
}
