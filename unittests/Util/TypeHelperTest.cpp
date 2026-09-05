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

TEST_F(TypeHelperTests, test_arrayTypesUnify_withWildcard_1) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, ShapedType::kDynamic});
  ArrayType b = ArrayType::get(tyIndex, {2, 5});
  ASSERT_TRUE(arrayTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_arrayTypesUnify_withWildcard_2) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, ShapedType::kDynamic});
  ArrayType b = ArrayType::get(tyIndex, {ShapedType::kDynamic, 5});
  ASSERT_TRUE(arrayTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_arrayWildcardDimension) {
  felt::FeltType feltTy = felt::FeltType::get(&ctx);
  ArrayType wildcardArray = ArrayType::get(feltTy, {ShapedType::kDynamic});
  ArrayType staticArray = ArrayType::get(feltTy, {1});

  ASSERT_TRUE(typesUnify(wildcardArray, staticArray));
  ASSERT_TRUE(isMoreConcreteUnification(wildcardArray, staticArray));
  ASSERT_FALSE(isMoreConcreteUnification(staticArray, wildcardArray));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_arrayWildcardDimension3) {
  felt::FeltType feltTy = felt::FeltType::get(&ctx);
  ArrayType wildcardArray = ArrayType::get(feltTy, {4, 5, ShapedType::kDynamic});
  ArrayType staticArray = ArrayType::get(feltTy, {4, 5, 6});

  ASSERT_TRUE(typesUnify(wildcardArray, staticArray));
  ASSERT_TRUE(isMoreConcreteUnification(wildcardArray, staticArray));
  ASSERT_FALSE(isMoreConcreteUnification(staticArray, wildcardArray));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_arrayWildcardDimensionsMix) {
  felt::FeltType feltTy = felt::FeltType::get(&ctx);
  ArrayType firstArray = ArrayType::get(feltTy, {ShapedType::kDynamic, 2});
  ArrayType secondArray = ArrayType::get(feltTy, {2, ShapedType::kDynamic});

  ASSERT_TRUE(typesUnify(firstArray, secondArray));
  ASSERT_FALSE(isMoreConcreteUnification(firstArray, secondArray));
  ASSERT_FALSE(isMoreConcreteUnification(secondArray, firstArray));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_arrayWildcardAndSymbolRefDimension) {
  felt::FeltType feltTy = felt::FeltType::get(&ctx);
  ArrayType wildcardArray = ArrayType::get(feltTy, {ShapedType::kDynamic});
  ArrayType symbolRefArray = ArrayType::get(feltTy, {FlatSymbolRefAttr::get(&ctx, "N")});

  ASSERT_TRUE(typesUnify(wildcardArray, symbolRefArray));
  ASSERT_TRUE(isMoreConcreteUnification(wildcardArray, symbolRefArray));
  ASSERT_FALSE(isMoreConcreteUnification(symbolRefArray, wildcardArray));
}

TEST_F(TypeHelperTests, test_isMoreConcreteUnification_arrayWildcardAndAffineMapDimension) {
  felt::FeltType feltTy = felt::FeltType::get(&ctx);
  ArrayType wildcardArray = ArrayType::get(feltTy, {ShapedType::kDynamic});
  auto aff = AffineMapAttr::get(OpBuilder(&ctx).getDimIdentityMap());
  ArrayType affineMapArray = ArrayType::get(feltTy, {aff});

  ASSERT_TRUE(typesUnify(wildcardArray, affineMapArray));
  ASSERT_TRUE(isMoreConcreteUnification(wildcardArray, affineMapArray));
  ASSERT_FALSE(isMoreConcreteUnification(affineMapArray, wildcardArray));
}

TEST_F(TypeHelperTests, test_structTypesUnify) {
  // StructType itself cannot be created with `?` in its parameter list.
  // Instead, test that a nested ArrayType as a TypeAttr in the list passes.
  IndexType tyIndex = IndexType::get(&ctx);
  Attribute t1 = TypeAttr::get(ArrayType::get(tyIndex, {242, ShapedType::kDynamic}));
  Attribute t2 = TypeAttr::get(ArrayType::get(tyIndex, {ShapedType::kDynamic, 5}));
  StructType a = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {t1});
  StructType b = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {t2});
  ASSERT_TRUE(structTypesUnify(a, b));
}

TEST_F(TypeHelperTests, test_typesUnify_equalRecursiveTypesRespectRhsPrefix) {
  SymbolRefAttr targetBoxName = SymbolRefAttr::get(
      &ctx, "Target", ArrayRef<FlatSymbolRefAttr> {FlatSymbolRefAttr::get(&ctx, "Box")}
  );
  SymbolRefAttr includedBoxName = SymbolRefAttr::get(
      &ctx, "Lib",
      ArrayRef<FlatSymbolRefAttr> {
          FlatSymbolRefAttr::get(&ctx, "Target"), FlatSymbolRefAttr::get(&ctx, "Box")
      }
  );
  StructType targetBox = StructType::get(targetBoxName);
  StructType includedBox = StructType::get(includedBoxName);
  SymbolRefAttr targetHolderName = SymbolRefAttr::get(
      &ctx, "Target", ArrayRef<FlatSymbolRefAttr> {FlatSymbolRefAttr::get(&ctx, "Holder")}
  );
  SymbolRefAttr includedHolderName = SymbolRefAttr::get(
      &ctx, "Lib",
      ArrayRef<FlatSymbolRefAttr> {
          FlatSymbolRefAttr::get(&ctx, "Target"), FlatSymbolRefAttr::get(&ctx, "Holder")
      }
  );
  StructType targetHolder =
      StructType::get(targetHolderName, ArrayRef<Attribute> {TypeAttr::get(targetBox)});
  StructType includedHolder =
      StructType::get(includedHolderName, ArrayRef<Attribute> {TypeAttr::get(includedBox)});
  StructType collidingHolder =
      StructType::get(includedHolderName, ArrayRef<Attribute> {TypeAttr::get(targetBox)});
  ArrayType targetArray = ArrayType::get(targetBox, {2});
  PodType targetPod = PodType::get(
      &ctx, ArrayRef {RecordAttr::get(&ctx, StringAttr::get(&ctx, "value"), targetBox)}
  );
  FunctionType targetFunction = FunctionType::get(&ctx, {IndexType::get(&ctx)}, {targetBox});
  llvm::StringRef includedNamespace = "Lib";
  ArrayRef<llvm::StringRef> rhsPrefix(&includedNamespace, 1);

  EXPECT_TRUE(typesUnify(includedBox, targetBox, rhsPrefix));
  EXPECT_TRUE(typesUnify(includedHolder, targetHolder, rhsPrefix));
  EXPECT_FALSE(typesUnify(targetBox, targetBox, rhsPrefix));
  EXPECT_FALSE(typesUnify(collidingHolder, targetHolder, rhsPrefix));
  EXPECT_FALSE(typesUnify(targetArray, targetArray, rhsPrefix));
  EXPECT_FALSE(typesUnify(targetPod, targetPod, rhsPrefix));
  EXPECT_FALSE(typesUnify(targetFunction, targetFunction, rhsPrefix));
  EXPECT_TRUE(typesUnify(IndexType::get(&ctx), IndexType::get(&ctx), rhsPrefix));
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

TEST_F(TypeHelperTests, test_functionTypesUnify_equalSymbolsDoNotChangeGenericUnifications) {
  FlatSymbolRefAttr param = FlatSymbolRefAttr::get(&ctx, "F");
  StructType structType =
      StructType::get(FlatSymbolRefAttr::get(&ctx, "Box"), ArrayRef<Attribute> {param});
  FunctionType functionType = FunctionType::get(&ctx, {structType}, {});
  UnificationMap unifications;

  ASSERT_TRUE(functionTypesUnify(functionType, functionType, {}, &unifications));
  EXPECT_TRUE(unifications.empty());
}

TEST_F(TypeHelperTests, test_functionTypesUnify_recordsRepeatedCandidates) {
  FlatSymbolRefAttr param = FlatSymbolRefAttr::get(&ctx, "F");
  FlatSymbolRefAttr global = FlatSymbolRefAttr::get(&ctx, "G");
  FeltConstAttr literal = FeltConstAttr::get(&ctx, APInt(8, 35), FeltType::get(&ctx, "bn128"));
  FlatSymbolRefAttr box = FlatSymbolRefAttr::get(&ctx, "Box");
  StructType globalType = StructType::get(box, ArrayRef<Attribute> {global});
  StructType literalType = StructType::get(box, ArrayRef<Attribute> {literal});
  StructType parameterType = StructType::get(box, ArrayRef<Attribute> {param});
  FunctionType caller = FunctionType::get(&ctx, {globalType, literalType}, {});
  FunctionType callee = FunctionType::get(&ctx, {parameterType, parameterType}, {});
  UnificationMap unifications;
  llvm::DenseMap<std::pair<SymbolRefAttr, Side>, SmallVector<Attribute, 2>> candidates;
  auto recordCandidate = [&](SymbolRefAttr symbol, Side side, Attribute value) {
    candidates[{symbol, side}].push_back(value);
  };

  ASSERT_TRUE(functionTypesUnify(caller, callee, {}, &unifications, recordCandidate));
  auto key = std::make_pair(param, Side::RHS);
  ASSERT_TRUE(unifications.contains(key));
  EXPECT_FALSE(unifications.lookup(key));
  ASSERT_EQ(candidates.lookup(key).size(), 2);
  EXPECT_EQ(candidates.lookup(key)[0], global);
  EXPECT_EQ(candidates.lookup(key)[1], literal);
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
