//===-- SMTAttributeTests.cpp - Unit tests for SMT attributes ---*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include <llvm/ADT/APInt.h>

class SMTAttributeTests : public LLZKTest {};

TEST_F(SMTAttributeTests, NumericAPIntStorageReusesEqualValuesAcrossWidths) {
  auto expectStorageReuse = [&](const llvm::APInt &narrow, const llvm::APInt &wide) {
    ASSERT_TRUE(llvm::APInt::isSameValue(narrow, wide));
    EXPECT_EQ(
        llvm::hash_combine(llzk::DynamicAPIntValue(llzk::toDynamicAPInt(narrow))),
        llvm::hash_combine(llzk::DynamicAPIntValue(llzk::toDynamicAPInt(wide)))
    );

    llzk::felt::FeltConstAttr narrowAttr =
        llzk::felt::FeltConstAttr::get(&ctx, llzk::toDynamicAPInt(narrow));
    llzk::felt::FeltConstAttr wideAttr =
        llzk::felt::FeltConstAttr::get(&ctx, llzk::toDynamicAPInt(wide));
    EXPECT_EQ(narrowAttr, wideAttr);
  };

  expectStorageReuse(llvm::APInt(1, 0), llvm::APInt(128, 0));

  llvm::APInt multiword = llvm::APInt::getOneBitSet(65, 64) | llvm::APInt(65, 7);
  expectStorageReuse(multiword, multiword.zext(129));
}

TEST_F(SMTAttributeTests, DynamicIntegerStorageUsesSignedNumericIdentity) {
  for (int64_t value : {-3, -1, 0, 1, 7}) {
    llvm::DynamicAPInt small(value);
    llvm::DynamicAPInt wide(llvm::APInt(256, value, true));
    auto a = llzk::felt::FeltConstAttr::get(&ctx, small);
    auto b = llzk::felt::FeltConstAttr::get(&ctx, wide);
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.getValue(), small);
    EXPECT_EQ(
        llvm::hash_combine(llzk::DynamicAPIntValue(small)),
        llvm::hash_combine(llzk::DynamicAPIntValue(wide))
    );
  }
  auto negative = llzk::felt::FeltConstAttr::get(&ctx, llvm::DynamicAPInt(-1));
  auto positive =
      llzk::felt::FeltConstAttr::get(&ctx, llzk::toDynamicAPInt("18446744073709551615"));
  EXPECT_NE(negative, positive);
}

TEST_F(SMTAttributeTests, BuiltinIntegerConversionPreservesSignedness) {
  auto decode = [&](mlir::IntegerType type, uint64_t bits) {
    return llzk::integerAttrToDynamicAPInt(mlir::IntegerAttr::get(type, bits));
  };
  EXPECT_EQ(
      decode(mlir::IntegerType::get(&ctx, 8, mlir::IntegerType::Unsigned), 255),
      llvm::DynamicAPInt(255)
  );
  EXPECT_EQ(decode(mlir::IntegerType::get(&ctx, 8), 255), llvm::DynamicAPInt(-1));
  EXPECT_EQ(decode(mlir::IntegerType::get(&ctx, 1), 1), llvm::DynamicAPInt(1));
  EXPECT_EQ(
      decode(mlir::IntegerType::get(&ctx, 1, mlir::IntegerType::Signed), 1), llvm::DynamicAPInt(-1)
  );
}
