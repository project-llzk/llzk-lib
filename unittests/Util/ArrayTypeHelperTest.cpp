//===-- ArrayTypeHelperTest.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Util/Debug.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/APInt.h>

#include <gtest/gtest.h>

using namespace llzk;
using namespace llzk::array;
using namespace mlir;

class ArrayTypeHelperTests : public LLZKTest {
protected:
  ArrayTypeHelperTests() : LLZKTest() {}
};

TEST_F(ArrayTypeHelperTests, test_delinearize_too_small) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {2, 4});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  std::optional<SmallVector<Attribute>> r = idxGen.delinearize(-1, &ctx);
  ASSERT_FALSE(r.has_value());
}

TEST_F(ArrayTypeHelperTests, test_delinearize_too_big) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {2, 4});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  ASSERT_EQ(ty.getNumElements(), 8);

  std::optional<SmallVector<Attribute>> x = idxGen.delinearize(8, &ctx);
  ASSERT_FALSE(x.has_value());
}

TEST_F(ArrayTypeHelperTests, test_linearize_too_small) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {2, 4});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  SmallVector<int64_t> multiDimIdx({0, -1});
  std::optional<int64_t> r = idxGen.linearize(ArrayRef(multiDimIdx));
  ASSERT_FALSE(r.has_value());
}

TEST_F(ArrayTypeHelperTests, test_linearize_too_big) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {2, 4});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  SmallVector<int64_t> multiDimIdx({5, 5});
  std::optional<int64_t> r = idxGen.linearize(ArrayRef(multiDimIdx));
  ASSERT_FALSE(r.has_value());
}

#ifndef NDEBUG
TEST_F(ArrayTypeHelperTests, test_linearize_too_few_dims) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {2, 4});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  EXPECT_DEATH(
      {
        SmallVector<int64_t> multiDimIdx({0});
        idxGen.linearize(ArrayRef(multiDimIdx));
      },
      "" // Expect assert failure within IndexingUtils.cpp but there's no message
  );
}

TEST_F(ArrayTypeHelperTests, test_linearize_too_many_dims) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {2, 4});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  EXPECT_DEATH(
      {
        SmallVector<int64_t> multiDimIdx({0, 0, 0});
        idxGen.linearize(ArrayRef(multiDimIdx));
      },
      "" // Expect assert failure within IndexingUtils.cpp but there's no message
  );
}
#endif

// Demonstrate that `toI64` in `ArrayTypeHelper.cpp` should use `trySExtValue()` rather
// than `getSExtValue()` to avoid asserting when the value does not fit in int64_t.
TEST_F(ArrayTypeHelperTests, test_linearize_index_wider_than_64bits_returns_nullopt) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {10});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  mlir::OwningOpRef<mlir::ModuleOp> moduleOp = mlir::ModuleOp::create(loc);
  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(moduleOp->getBody());

  llvm::APInt bigVal(128, 0);
  bigVal.setBit(65); // 2^65 does not fit in int64_t
  auto bigConst = builder.create<mlir::arith::ConstantOp>(
      loc, mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 128), bigVal)
  );

  SmallVector<Value> indices = {bigConst.getResult()};
  std::optional<int64_t> r = idxGen.linearize(ArrayRef<Value>(indices));
  ASSERT_FALSE(r.has_value());
}

// Another test demonstrating that `toI64` in `ArrayTypeHelper.cpp` should use `trySExtValue()`.
TEST_F(ArrayTypeHelperTests, test_check_and_convert_index_wider_than_64bits_returns_nullopt) {
  ArrayType ty = ArrayType::get(IndexType::get(&ctx), {10});
  ArrayIndexGen idxGen = ArrayIndexGen::from(ty);

  mlir::OwningOpRef<mlir::ModuleOp> moduleOp = mlir::ModuleOp::create(loc);
  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(moduleOp->getBody());

  llvm::APInt bigVal(128, 0);
  bigVal.setBit(65); // 2^65 does not fit in int64_t
  auto bigConst = builder.create<mlir::arith::ConstantOp>(
      loc, mlir::IntegerAttr::get(mlir::IntegerType::get(&ctx, 128), bigVal)
  );

  // Build an op containing bigConst, providing an OperandRange for checkAndConvert()
  auto arrCreate = builder.create<CreateArrayOp>(loc, ty, ValueRange {});
  auto readOp =
      builder.create<ReadArrayOp>(loc, arrCreate.getResult(), ValueRange {bigConst.getResult()});

  std::optional<SmallVector<Attribute>> r = idxGen.checkAndConvert(readOp.getIndices());
  ASSERT_FALSE(r.has_value());
}
