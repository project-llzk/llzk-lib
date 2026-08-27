//===-- TypeHelperTest.cpp - Unit tests for symbol utilities ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/TypeHelper.h"

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"

#include <mlir/IR/BuiltinTypeInterfaces.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::pod;

class TypeHelperTests : public LLZKTest {
protected:
  TypeHelperTests() : LLZKTest(), errFn([this]() { return InFlightDiagnosticWrapper(&ctx); }) {}

  OwningEmitErrorFn errFn;
};

TEST_F(TypeHelperTests, test_arrayTypesUnify_withDynamic_1) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, ShapedType::kDynamic});
  ArrayType b = ArrayType::get(tyIndex, {2, 5});
  ASSERT_TRUE(arrayTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_arrayTypesUnify_withDynamic_2) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, ShapedType::kDynamic});
  ArrayType b = ArrayType::get(tyIndex, {ShapedType::kDynamic, 5});
  ASSERT_TRUE(arrayTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_structTypesUnify) {
  IndexType tyIndex = IndexType::get(&ctx);
  Attribute i1 = IntegerAttr::get(tyIndex, 128);
  Attribute i2 = IntegerAttr::get(tyIndex, ShapedType::kDynamic);
  StructType a = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {i1});
  StructType b = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {i2});
  // `false` because StructType does not allow `kDynamic`
  ASSERT_FALSE(structTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_podTypesUnify_Pass) {
  IndexType tyIndex = IndexType::get(&ctx);
  auto r1 = RecordAttr::get(&ctx, StringAttr::get(&ctx, "r"), tyIndex);
  auto r2 = RecordAttr::get(&ctx, StringAttr::get(&ctx, "r"), tyIndex);
  PodType a = PodType::get(&ctx, ArrayRef {r1});
  PodType b = PodType::get(&ctx, ArrayRef {r2});
  ASSERT_TRUE(podTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_podTypesUnify_Name_Fail) {
  IndexType tyIndex = IndexType::get(&ctx);
  auto r1 = RecordAttr::get(&ctx, StringAttr::get(&ctx, "r"), tyIndex);
  auto r2 = RecordAttr::get(&ctx, StringAttr::get(&ctx, "q"), tyIndex);
  PodType a = PodType::get(&ctx, ArrayRef {r1});
  PodType b = PodType::get(&ctx, ArrayRef {r2});
  ASSERT_FALSE(podTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_podTypesUnify_Type_Fail) {
  auto r1 = RecordAttr::get(&ctx, StringAttr::get(&ctx, "r"), IndexType::get(&ctx));
  auto r2 = RecordAttr::get(&ctx, StringAttr::get(&ctx, "r"), IntegerType::get(&ctx, 8));
  PodType a = PodType::get(&ctx, ArrayRef {r1});
  PodType b = PodType::get(&ctx, ArrayRef {r2});
  ASSERT_FALSE(podTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_functionTypesUnify_Pass) {
  IndexType tyIndex = IndexType::get(&ctx);
  FunctionType a = FunctionType::get(&ctx, {tyIndex}, {tyIndex});
  FunctionType b = FunctionType::get(&ctx, {tyIndex}, {tyIndex});
  ASSERT_TRUE(functionTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_functionTypesUnify_Input_Fail) {
  IndexType tyIndex = IndexType::get(&ctx);
  FunctionType a = FunctionType::get(&ctx, {IntegerType::get(&ctx, 8)}, {tyIndex});
  FunctionType b = FunctionType::get(&ctx, {tyIndex}, {tyIndex});
  ASSERT_FALSE(functionTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_functionTypesUnify_Output_Fail) {
  IndexType tyIndex = IndexType::get(&ctx);
  FunctionType a = FunctionType::get(&ctx, {tyIndex}, {IntegerType::get(&ctx, 8)});
  FunctionType b = FunctionType::get(&ctx, {tyIndex}, {tyIndex});
  ASSERT_FALSE(functionTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_functionTypesUnifyWithCommonFeltField) {
  FeltType unspecified = FeltType::get(&ctx);
  FeltType bn128 = FeltType::get(&ctx, "bn128");
  FeltType babybear = FeltType::get(&ctx, "babybear");

  FunctionType callee = FunctionType::get(&ctx, {unspecified}, {unspecified});
  ASSERT_TRUE(
      functionTypesUnifyWithCommonFeltField(FunctionType::get(&ctx, {bn128}, {bn128}), callee)
  );
  ASSERT_FALSE(
      functionTypesUnifyWithCommonFeltField(FunctionType::get(&ctx, {bn128}, {babybear}), callee)
  );

  // A callee with explicit fields describes a cross-field conversion and is not subject to the
  // shared substitution for unspecified felt types.
  FunctionType conversion = FunctionType::get(&ctx, {bn128}, {babybear});
  ASSERT_TRUE(functionTypesUnifyWithCommonFeltField(
      FunctionType::get(&ctx, {bn128}, {babybear}), conversion
  ));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_feltField) {
  FeltType unspecified = FeltType::get(&ctx);
  FeltType specified = FeltType::get(&ctx, "bn128");

  ASSERT_TRUE(typesUnify(unspecified, specified));
  ASSERT_TRUE(typesUnify(specified, unspecified));
  ASSERT_TRUE(isMoreConcreteUnification(unspecified, specified));
  ASSERT_FALSE(isMoreConcreteUnification(specified, unspecified));
}

TEST_F(TypeHelperTests, test_typesUnifyWithoutLosingFeltFields) {
  FeltType unspecified = FeltType::get(&ctx);
  FeltType specified = FeltType::get(&ctx, "bn128");
  ArrayType unspecifiedArray = ArrayType::get(unspecified, {2});
  ArrayType specifiedArray = ArrayType::get(specified, {2});

  ASSERT_TRUE(typesUnifyWithoutLosingFeltFields(unspecified, unspecified));
  ASSERT_TRUE(typesUnifyWithoutLosingFeltFields(unspecified, specified));
  ASSERT_TRUE(typesUnifyWithoutLosingFeltFields(specified, specified));
  ASSERT_FALSE(typesUnifyWithoutLosingFeltFields(specified, unspecified));
  ASSERT_FALSE(typesUnifyWithoutLosingFeltFields(specifiedArray, unspecifiedArray));
  ASSERT_TRUE(typesUnifyWithoutLosingFeltFields(specified, unspecified, {}, nullptr, Side::RHS));
  ASSERT_FALSE(typesUnifyWithoutLosingFeltFields(unspecified, specified, {}, nullptr, Side::RHS));
}

TEST_F(TypeHelperTests, test_typesHaveConflictingFeltFields_functionsIgnoreOtherTypeMismatch) {
  FunctionType bn128 =
      FunctionType::get(&ctx, {IndexType::get(&ctx), FeltType::get(&ctx, "bn128")}, {});
  FunctionType babybear =
      FunctionType::get(&ctx, {IntegerType::get(&ctx, 1), FeltType::get(&ctx, "babybear")}, {});

  ASSERT_FALSE(typesUnify(bn128, babybear));
  ASSERT_TRUE(typesHaveConflictingFeltFields(bn128, babybear));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_nestedFeltField) {
  FeltType unspecified = FeltType::get(&ctx);
  FeltType specified = FeltType::get(&ctx, "bn128");
  ArrayType oldTy = ArrayType::get(specified, {2});
  ArrayType newTy = ArrayType::get(unspecified, {2});

  ASSERT_TRUE(typesUnify(oldTy, newTy));
  ASSERT_FALSE(isMoreConcreteUnification(oldTy, newTy));
}

TEST_F(TypeHelperTests, test_forceIntToIndexType_fromI1) {
  // create a boolean IntegerAttr
  IntegerAttr a = IntegerAttr::get(IntegerType::get(&ctx, 1), 1);
  // Force IndexType on it without changing the value
  FailureOr<IntegerAttr> b = forceIntType(a, errFn);
  ASSERT_TRUE(succeeded(b));
  ASSERT_TRUE(llvm::isa<IndexType>(b->getType()));
  ASSERT_EQ(b->getValue().getBitWidth(), IndexType::kInternalStorageBitWidth);
  ASSERT_EQ(b->getValue(), APInt(IndexType::kInternalStorageBitWidth, 1));
}

TEST_F(TypeHelperTests, test_forceIntToIndexType_fromI8) {
  // create an 8-bit IntegerAttr
  IntegerAttr a = IntegerAttr::get(IntegerType::get(&ctx, 8), 42);
  // Force IndexType on it without changing the value
  FailureOr<IntegerAttr> b = forceIntType(a, errFn);
  ASSERT_TRUE(succeeded(b));
  ASSERT_TRUE(llvm::isa<IndexType>(b->getType()));
  ASSERT_EQ(b->getValue().getBitWidth(), IndexType::kInternalStorageBitWidth);
  ASSERT_EQ(b->getValue(), APInt(IndexType::kInternalStorageBitWidth, 42));
}

TEST_F(TypeHelperTests, test_forceIntToIndexType_fromI256) {
  // create an 256-bit IntegerAttr with larger value than IndexType can hold
  APInt bigValue = APInt::getMaxValue(256);
  IntegerAttr a = IntegerAttr::get(IntegerType::get(&ctx, 256), bigValue);
  // Force IndexType on it without changing the value
  ASSERT_DEATH(
      {
        if (failed(forceIntType(a, errFn))) {
          std::abort();
        }
      },
      "error: value is too large for `index` type: -1"
  );
}
