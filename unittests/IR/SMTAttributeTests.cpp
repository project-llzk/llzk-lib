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
        llvm::hash_combine(llzk::APIntValue(narrow)), llvm::hash_combine(llzk::APIntValue(wide))
    );

    llzk::felt::FeltConstAttr narrowAttr = llzk::felt::FeltConstAttr::get(&ctx, narrow);
    llzk::felt::FeltConstAttr wideAttr = llzk::felt::FeltConstAttr::get(&ctx, wide);
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

TEST_F(SMTAttributeTests, LimbBuildersPreserveUnsignedValuesAndFieldTypes) {
  using namespace llzk::felt;
  const uint64_t parts[] = {7, UINT64_C(0x8000000000000000)};
  const auto expected = llzk::toDynamicAPInt("170141183460469231731687303715884105735");
  auto type = FeltType::get(&ctx, "babybear");
  auto explicitType = FeltConstAttr::get(&ctx, parts, type);
  auto namedField = FeltConstAttr::get(&ctx, parts, "babybear");
  auto unspecified = FeltConstAttr::get(&ctx, parts);
  EXPECT_EQ(explicitType, namedField);
  EXPECT_EQ(explicitType.getType(), type);
  EXPECT_EQ(unspecified.getType(), FeltType::get(&ctx));
  EXPECT_EQ(explicitType.getValue(), expected);
  EXPECT_EQ(unspecified.getValue(), expected);

  auto name = mlir::StringAttr::get(&ctx, "custom");
  auto spec = FieldSpecAttr::get(&ctx, name, parts);
  EXPECT_EQ(spec, FieldSpecAttr::get(&ctx, "custom", parts));
  EXPECT_EQ(spec.getFieldName(), name);
  EXPECT_EQ(spec.getPrime(), expected);
}

TEST_F(SMTAttributeTests, LimbBuildersCanonicalizeEmptyAndHighZeroLimbs) {
  using namespace llzk::felt;
  llvm::ArrayRef<uint64_t> empty;
  const uint64_t zero[] = {0, 0};
  EXPECT_EQ(llzk::toDynamicAPInt(empty), 0);
  EXPECT_EQ(FeltConstAttr::get(&ctx, empty), FeltConstAttr::get(&ctx, zero));
  EXPECT_EQ(FeltConstAttr::get(&ctx, empty).getValue(), 0);

  const uint64_t parts[] = {UINT64_MAX};
  const uint64_t padded[] = {UINT64_MAX, 0, 0};
  EXPECT_EQ(FeltConstAttr::get(&ctx, parts), FeltConstAttr::get(&ctx, padded));
  EXPECT_EQ(
      FeltConstAttr::get(&ctx, parts).getValue(), llzk::toDynamicAPInt("18446744073709551615")
  );
  EXPECT_EQ(FieldSpecAttr::get(&ctx, "custom", parts), FieldSpecAttr::get(&ctx, "custom", padded));
}
