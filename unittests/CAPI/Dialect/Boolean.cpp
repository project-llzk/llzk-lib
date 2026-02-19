//===-- Boolean.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Bool.h"

#include <mlir-c/BuiltinAttributes.h>

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_boolean) {
  (void)mlirGetDialectHandle__llzk__boolean__();
}

class CmpAttrTest : public CAPITest, public testing::WithParamInterface<LlzkCmp> {};

TEST_P(CmpAttrTest, llzk_felt_cmp_predicate_attr_get) {
  auto attr = llzkBool_FeltCmpPredicateAttrGet(context, GetParam());
  EXPECT_NE(attr.ptr, (void *)NULL);
}

INSTANTIATE_TEST_SUITE_P(
    AllLlzkCmpValues, CmpAttrTest,
    testing::Values(LlzkCmp_EQ, LlzkCmp_LE, LlzkCmp_LT, LlzkCmp_GE, LlzkCmp_GT, LlzkCmp_NE)
);

TEST_F(CAPITest, llzk_attribute_is_a_felt_cmp_predicate_attr_pass) {
  auto attr = llzkBool_FeltCmpPredicateAttrGet(context, LlzkCmp_EQ);
  EXPECT_TRUE(llzkAttributeIsA_Bool_FeltCmpPredicateAttr(attr));
}

TEST_F(CAPITest, llzk_attribute_is_a_felt_cmp_predicate_attr_fail) {
  auto attr = mlirUnitAttrGet(context);
  EXPECT_TRUE(!llzkAttributeIsA_Bool_FeltCmpPredicateAttr(attr));
}
