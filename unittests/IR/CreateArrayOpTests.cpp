//===-- CreateArrayOpTests.cpp - Unit tests for CreateArrayOp ---*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include "OpTestBase.h"

using namespace mlir;
using namespace llzk;
using namespace llzk::array;
using namespace llzk::felt;

//===------------------------------------------------------------------===//
// CreateArrayOp::build(..., ArrayType, ValueRange)
//===------------------------------------------------------------------===//

TEST_F(OpTests, testElementInit_GoodEmpty) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {2, 2}); // !array.type<2,2 x index>
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy);
  ASSERT_TRUE(verify(op));
}

TEST_F(OpTests, testElementInit_GoodNonEmpty) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {2}); // !array.type<2 x index>
  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 766);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 562);
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange {v1, v2});
  ASSERT_TRUE(verify(op));
}

TEST_F(OpTests, testElementInit_TooFew) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {5}); // !array.type<5 x index>
  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 766);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 562);
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange {v1, v2});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op failed to verify that operand types match result type"
  );
}

TEST_F(OpTests, testElementInit_TooMany) {
  OpBuilder bldr(mod->getRegion());
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {1}); // !array.type<1 x index>
  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 766);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 562);
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy, ValueRange {v1, v2});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op failed to verify that operand types match result type"
  );
}

TEST_F(OpTests, testElementInit_WithAffineMapType) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>
  CreateArrayOp op = bldr.create<CreateArrayOp>(loc, arrTy);
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(0\\) does not match the number "
      "of affine map instantiations \\(1\\) required by the type"
  );
}

//===------------------------------------------------------------------===//
// CreateArrayOp::build(..., ArrayType, ArrayRef<ValueRange>, ArrayRef<int32_t>)
//===------------------------------------------------------------------===//

TEST_F(OpTests, testMapOpInit_Good) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !array.type<#m,#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 98);
  CreateArrayOp op = bldr.create<CreateArrayOp>(
      loc, arrTy, ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef {1, 1}
  );
  ASSERT_TRUE(verify(op));
}

TEST_F(OpTests, testMapOpInit_Op1_Dim1_Type2) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !array.type<#m,#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  CreateArrayOp op =
      bldr.create<CreateArrayOp>(loc, arrTy, ArrayRef {ValueRange {v1}}, ArrayRef<int32_t> {1});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(1\\) does not match the number "
      "of affine map instantiations \\(2\\) required by the type"
  );
}

TEST_F(OpTests, testMapOpInit_Op1_Dim2_Type2) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !array.type<#m,#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  CreateArrayOp op =
      bldr.create<CreateArrayOp>(loc, arrTy, ArrayRef {ValueRange {v1}}, ArrayRef<int32_t> {1, 0});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(1\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(2\\)"
  );
}

TEST_F(OpTests, testMapOpInit_Op2_Dim1_Type2) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m, m});  // !array.type<#m,#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 98);
  CreateArrayOp op = bldr.create<CreateArrayOp>(
      loc, arrTy, ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef<int32_t> {1}
  );
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(2\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(1\\)"
  );
}

TEST_F(OpTests, testMapOpInit_Op3_Dim3_Type1) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 98);
  auto v3 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CreateArrayOp op = bldr.create<CreateArrayOp>(
      loc, arrTy, ArrayRef {ValueRange {v1}, ValueRange {v2}, ValueRange {v3}},
      ArrayRef<int32_t> {1, 1, 1}
  );
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(3\\) does not match the number "
      "of affine map instantiations \\(1\\) required by the type"
  );
}

TEST_F(OpTests, testMapOpInit_Op3_Dim2_Type1) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 98);
  auto v3 = bldr.create<arith::ConstantIndexOp>(loc, 4);
  CreateArrayOp op = bldr.create<CreateArrayOp>(
      loc, arrTy, ArrayRef {ValueRange {v1}, ValueRange {v2}, ValueRange {v3}},
      ArrayRef<int32_t> {1, 1}
  );
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(3\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(2\\)"
  );
}

TEST_F(OpTests, testMapOpInit_Op2_Dim3_Type1) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 98);
  CreateArrayOp op = bldr.create<CreateArrayOp>(
      loc, arrTy, ArrayRef {ValueRange {v1}, ValueRange {v2}}, ArrayRef<int32_t> {1, 1, 1}
  );
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op map instantiation group count \\(2\\) does not match with length "
      "of 'mapOpGroupSizes' attribute \\(3\\)"
  );
}

TEST_F(OpTests, testMapOpInit_NumDimsTooHigh) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  CreateArrayOp op =
      bldr.create<CreateArrayOp>(loc, arrTy, ArrayRef {ValueRange {v1}}, ArrayRef<int32_t> {9});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op instantiation of map 0 expected 1 but found 9 dimension values "
      "in \\(\\)"
  );
}

TEST_F(OpTests, testMapOpInit_TooManyOpsForMap) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  auto v2 = bldr.create<arith::ConstantIndexOp>(loc, 23);
  CreateArrayOp op =
      bldr.create<CreateArrayOp>(loc, arrTy, ArrayRef {ValueRange {v1, v2}}, ArrayRef<int32_t> {1});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op instantiation of map 0 expected 0 but found 1 symbol values in "
      "\\[\\]"
  );
}

TEST_F(OpTests, testMapOpInit_TooFewOpsForMap) {
  OpBuilder bldr(mod->getRegion());
  // (d0, d1) -> (d0 + d1)
  AffineMapAttr m = AffineMapAttr::get(
      AffineMap::get(
          /*dimCount=*/2, /*symbolCount=*/0, bldr.getAffineDimExpr(0) + bldr.getAffineDimExpr(1)
      )
  );
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m}); // !array.type<#m x index>

  auto v1 = bldr.create<arith::ConstantIndexOp>(loc, 10);
  CreateArrayOp op =
      bldr.create<CreateArrayOp>(loc, arrTy, ArrayRef {ValueRange {v1}}, ArrayRef<int32_t> {1});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op instantiation of map 0 expected 2 but found 1 dimension values "
      "in \\(\\)"
  );
}

TEST_F(OpTests, testMapOpInit_WrongTypeForMapOperands) {
  OpBuilder bldr(mod->getRegion());
  AffineMapAttr m = AffineMapAttr::get(bldr.getDimIdentityMap()); // (d0) -> (d0)
  ArrayType arrTy = ArrayType::get(bldr.getIndexType(), {m});     // !array.type<#m x index>

  FeltConstAttr a = bldr.getAttr<FeltConstAttr>(APInt::getZero(64));
  auto v1 = bldr.create<FeltConstantOp>(loc, a);
  CreateArrayOp op =
      bldr.create<CreateArrayOp>(loc, arrTy, ArrayRef {ValueRange {v1}}, ArrayRef<int32_t> {1});
  EXPECT_DEATH(
      { verifyOrDie(op); },
      "error: 'array.new' op operand #0 must be variadic of index, but got '!felt.type'"
  );
}
