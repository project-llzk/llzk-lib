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
#include "llzk/Dialect/Felt/IR/Attrs.h"
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

TEST_F(TypeHelperTests, test_templateParamTypeCompatibility_feltFields) {
  FeltType fieldless = FeltType::get(&ctx);
  FeltType bn128 = FeltType::get(&ctx, "bn128");
  FeltType goldilocks = FeltType::get(&ctx, "goldilocks");

  ASSERT_TRUE(isTemplateParamTypeCompatible(bn128, fieldless));
  ASSERT_TRUE(isTemplateParamTypeCompatible(fieldless, fieldless));
  ASSERT_FALSE(isTemplateParamTypeCompatible(fieldless, bn128));
  ASSERT_TRUE(isTemplateParamTypeCompatible(bn128, bn128));
  ASSERT_FALSE(isTemplateParamTypeCompatible(goldilocks, bn128));
  ASSERT_FALSE(isTemplateParamTypeCompatible(IndexType::get(&ctx), bn128));
  ASSERT_FALSE(isTemplateParamTypeCompatible(std::nullopt, bn128));
}

TEST_F(TypeHelperTests, test_templateParamValuesUnify_feltRepresentations) {
  FeltType fieldless = FeltType::get(&ctx);
  FeltType bn128 = FeltType::get(&ctx, "bn128");
  FeltType goldilocks = FeltType::get(&ctx, "goldilocks");
  FeltConstAttr unspecified = FeltConstAttr::get(&ctx, APInt(8, 35), fieldless);
  FeltConstAttr fielded = FeltConstAttr::get(&ctx, APInt(8, 35), bn128);
  FeltConstAttr differentValue = FeltConstAttr::get(&ctx, APInt(8, 36), bn128);
  FeltConstAttr differentField = FeltConstAttr::get(&ctx, APInt(8, 35), goldilocks);
  IntegerAttr integer = IntegerAttr::get(IndexType::get(&ctx), 35);
  FlatSymbolRefAttr actualSymbol = FlatSymbolRefAttr::get(&ctx, "Actual");
  FlatSymbolRefAttr inferredSymbol = FlatSymbolRefAttr::get(&ctx, "Inferred");

  EXPECT_TRUE(templateParamValuesUnify(unspecified, fielded, fieldless));
  EXPECT_TRUE(templateParamValuesUnify(unspecified, fielded, bn128));
  EXPECT_TRUE(templateParamValuesUnify(integer, fielded, bn128));
  EXPECT_TRUE(templateParamValuesUnify(fielded, integer, bn128));
  EXPECT_FALSE(templateParamValuesUnify(differentValue, fielded, bn128));
  EXPECT_FALSE(templateParamValuesUnify(differentField, fielded, fieldless));
  EXPECT_FALSE(templateParamValuesUnify(differentField, differentField, bn128));
  EXPECT_TRUE(templateParamValuesUnify(actualSymbol, inferredSymbol, bn128));
}

TEST_F(TypeHelperTests, test_templateParamValuesUnify_widthIndependentFeltValues) {
  static constexpr unsigned NARROW_WIDTH = 8;
  static constexpr unsigned WIDE_WIDTH = 64;
  static constexpr unsigned VALUE = 35;

  FeltType fieldless = FeltType::get(&ctx);
  FeltType bn128 = FeltType::get(&ctx, "bn128");
  FeltType goldilocks = FeltType::get(&ctx, "goldilocks");
  IntegerAttr wideInteger = IntegerAttr::get(IntegerType::get(&ctx, WIDE_WIDTH), VALUE);
  IntegerAttr differentWideInteger =
      IntegerAttr::get(IntegerType::get(&ctx, WIDE_WIDTH), VALUE + 1);
  FeltConstAttr narrowBn128 = FeltConstAttr::get(&ctx, APInt(NARROW_WIDTH, VALUE), bn128);
  FeltConstAttr wideBn128 = FeltConstAttr::get(&ctx, APInt(WIDE_WIDTH, VALUE), bn128);
  FeltConstAttr narrowGoldilocks = FeltConstAttr::get(&ctx, APInt(NARROW_WIDTH, VALUE), goldilocks);

  EXPECT_TRUE(templateParamValuesUnify(wideInteger, narrowBn128, fieldless));
  EXPECT_TRUE(templateParamValuesUnify(narrowBn128, wideInteger, fieldless));
  EXPECT_FALSE(templateParamValuesUnify(differentWideInteger, narrowBn128, fieldless));
  EXPECT_TRUE(templateParamValuesUnify(wideBn128, narrowBn128, fieldless));
  EXPECT_FALSE(templateParamValuesUnify(wideBn128, narrowGoldilocks, fieldless));
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
