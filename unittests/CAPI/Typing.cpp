//===-- Typing.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Typing.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include "CAPITestBase.h"

static bool test_callback1(MlirType, MlirType, void *) { return true; }

class TypingTest : public CAPITest {};

TEST_F(TypingTest, assert_valid_attr_for_param_of_type) {
  auto int_attr = createIndexAttribute(0);
  llzkAssertValidAttrForParamOfType(int_attr);
}

TEST_F(TypingTest, is_valid_type) { EXPECT_TRUE(llzkIsValidType(createIndexType())); }

TEST_F(TypingTest, is_valid_column_type) {
  auto null_op = MlirOperation {.ptr = NULL};
  EXPECT_TRUE(!llzkIsValidColumnType(createIndexType(), null_op));
}

TEST_F(TypingTest, is_valid_emit_eq_type) { EXPECT_TRUE(llzkIsValidEmitEqType(createIndexType())); }

TEST_F(TypingTest, is_valid_const_read_type) {
  EXPECT_TRUE(llzkIsValidConstReadType(createIndexType()));
}

TEST_F(TypingTest, is_valid_array_elem_type) {
  EXPECT_TRUE(llzkIsValidArrayElemType(createIndexType()));
}

TEST_F(TypingTest, is_valid_array_type) { EXPECT_TRUE(!llzkIsValidArrayType(createIndexType())); }

TEST_F(TypingTest, is_concrete_type) { EXPECT_TRUE(llzkIsConcreteType(createIndexType(), true)); }

TEST_F(TypingTest, has_affine_map_attr) { EXPECT_TRUE(!llzkHasAffineMapAttr(createIndexType())); }

TEST_F(TypingTest, type_params_unify_empty) { EXPECT_TRUE(llzkTypeParamsUnify(0, NULL, 0, NULL)); }

TEST_F(TypingTest, type_params_unify_pass) {
  auto string_ref = mlirStringRefCreateFromCString("N");

  MlirAttribute lhs[1] = {createIndexAttribute(0)};
  MlirAttribute rhs[1] = {mlirFlatSymbolRefAttrGet(mlirAttributeGetContext(lhs[0]), string_ref)};
  EXPECT_TRUE(llzkTypeParamsUnify(1, lhs, 1, rhs));
}

TEST_F(TypingTest, type_params_unify_fail) {
  MlirAttribute lhs[1] = {createIndexAttribute(0)};
  MlirAttribute rhs[1] = {createIndexAttribute(1)};
  EXPECT_TRUE(!llzkTypeParamsUnify(1, lhs, 1, rhs));
}

TEST_F(TypingTest, array_attr_type_params_unify_empty) {
  MlirAttribute lhs = mlirArrayAttrGet(context, 0, NULL);
  MlirAttribute rhs = mlirArrayAttrGet(context, 0, NULL);
  EXPECT_TRUE(llzkArrayAttrTypeParamsUnify(lhs, rhs));
}

TEST_F(TypingTest, array_attr_type_params_unify_pass) {
  auto string_ref = mlirStringRefCreateFromCString("N");

  MlirAttribute lhs[1] = {createIndexAttribute(0)};
  auto lhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(lhs[0]), 1, lhs);
  MlirAttribute rhs[1] = {mlirFlatSymbolRefAttrGet(mlirAttributeGetContext(*lhs), string_ref)};
  auto rhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(*lhs), 1, rhs);
  EXPECT_TRUE(llzkArrayAttrTypeParamsUnify(lhsAttr, rhsAttr));
}

TEST_F(TypingTest, array_attr_type_params_unify_fail) {
  MlirAttribute lhs[1] = {createIndexAttribute(0)};

  auto lhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(lhs[0]), 1, lhs);
  MlirAttribute rhs[1] = {createIndexAttribute(1)};

  auto rhsAttr = mlirArrayAttrGet(mlirAttributeGetContext(*lhs), 1, rhs);
  EXPECT_TRUE(!llzkArrayAttrTypeParamsUnify(lhsAttr, rhsAttr));
}

TEST_F(TypingTest, types_unify) {
  EXPECT_TRUE(llzkTypesUnify(createIndexType(), createIndexType(), 0, NULL));
}

TEST_F(TypingTest, is_more_concrete_unification) {
  EXPECT_TRUE(
      llzkIsMoreConcreteUnification(createIndexType(), createIndexType(), test_callback1, NULL)
  );
}

TEST_F(TypingTest, force_int_attr_type) {
  auto location = mlirLocationUnknownGet(context);
  auto in_attr = mlirIntegerAttrGet(mlirIntegerTypeGet(context, 64), 0);
  auto out_attr = llzkForceIntAttrType(in_attr, location);
  EXPECT_TRUE(!mlirAttributeEqual(in_attr, out_attr));
}

TEST_F(TypingTest, is_valid_global_type) { EXPECT_TRUE(llzkIsValidGlobalType(createIndexType())); }
