//===-- ArrayTypeHelperTest.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Util/Debug.h"

#include <gtest/gtest.h>

#include "../LLZKTestBase.h"

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
