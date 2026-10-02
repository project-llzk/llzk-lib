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

#include <mlir/IR/Diagnostics.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

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

TEST_F(SMTAttributeTests, LimbBuildersPreservePositiveValuesAndFieldTypes) {
  using namespace llzk::felt;
  const uint64_t parts[] = {7, UINT64_C(0x8000000000000000), 0};
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
  EXPECT_EQ(FeltConstAttr::get(&ctx, empty), FeltConstAttr::get(&ctx, zero));
  EXPECT_EQ(FeltConstAttr::get(&ctx, empty).getValue(), 0);

  const uint64_t parts[] = {UINT64_MAX, 0};
  const uint64_t padded[] = {UINT64_MAX, 0, 0};
  EXPECT_EQ(FeltConstAttr::get(&ctx, parts), FeltConstAttr::get(&ctx, padded));
  EXPECT_EQ(
      FeltConstAttr::get(&ctx, parts).getValue(), llzk::toDynamicAPInt("18446744073709551615")
  );
  EXPECT_EQ(FieldSpecAttr::get(&ctx, "custom", parts), FieldSpecAttr::get(&ctx, "custom", padded));
}

TEST_F(SMTAttributeTests, FeltLimbBuildersPreserveSignedValuesAndFieldTypes) {
  using namespace llzk::felt;
  struct TestCase {
    llvm::SmallVector<uint64_t> parts;
    const char *expected;
  };
  auto type = FeltType::get(&ctx, "babybear");
  for (const auto &test : {
           TestCase {{}, "0"},
           TestCase {{UINT64_MAX}, "-1"},
           TestCase {{UINT64_MAX, UINT64_MAX}, "-1"},
           TestCase {{uint64_t(1) << 63}, "-9223372036854775808"},
           TestCase {{0, uint64_t(1) << 63}, "-170141183460469231731687303715884105728"},
           TestCase {{UINT64_MAX, UINT64_MAX - 1}, "-18446744073709551617"},
           TestCase {{UINT64_MAX, 0}, "18446744073709551615"},
           TestCase {{2, 1}, "18446744073709551618"},
       }) {
    SCOPED_TRACE(test.expected);
    auto expected = llzk::toDynamicAPInt(test.expected);
    auto explicitType = FeltConstAttr::get(&ctx, test.parts, type);
    auto namedField = FeltConstAttr::get(&ctx, test.parts, "babybear");
    auto unspecified = FeltConstAttr::get(&ctx, test.parts);
    EXPECT_EQ(explicitType, namedField);
    EXPECT_EQ(explicitType.getType(), type);
    EXPECT_EQ(unspecified.getType(), FeltType::get(&ctx));
    EXPECT_EQ(explicitType.getValue(), expected);
    EXPECT_EQ(unspecified.getValue(), expected);
  }
}

TEST_F(SMTAttributeTests, CheckedFieldLimbBuildersRejectModuliBelowTwo) {
  using namespace llzk::felt;
  unsigned diagnostics = 0;
  mlir::ScopedDiagnosticHandler handler(&ctx, [&](mlir::Diagnostic &diag) {
    EXPECT_NE(diag.str().find("field modulus must be at least 2"), std::string::npos);
    ++diagnostics;
    return mlir::success();
  });
  auto name = mlir::StringAttr::get(&ctx, "custom");
  for (const auto &parts : {
           llvm::SmallVector<uint64_t> {},
           llvm::SmallVector<uint64_t> {0},
           llvm::SmallVector<uint64_t> {1, 0},
           llvm::SmallVector<uint64_t> {UINT64_MAX},
           llvm::SmallVector<uint64_t> {UINT64_MAX, UINT64_MAX},
           llvm::SmallVector<uint64_t> {0, uint64_t(1) << 63},
       }) {
    EXPECT_FALSE(FieldSpecAttr::getChecked(loc, &ctx, name, llvm::ArrayRef<uint64_t>(parts)));
    EXPECT_FALSE(
        FieldSpecAttr::getChecked(
            loc, &ctx, llvm::StringRef("custom"), llvm::ArrayRef<uint64_t>(parts)
        )
    );
  }
  EXPECT_EQ(diagnostics, 12U);
  for (const auto &parts : {
           llvm::SmallVector<uint64_t> {2},
           llvm::SmallVector<uint64_t> {2, 0},
           llvm::SmallVector<uint64_t> {UINT64_MAX, 0},
       }) {
    auto expected = FieldSpecAttr::get(&ctx, name, parts);
    EXPECT_EQ(
        FieldSpecAttr::getChecked(loc, &ctx, name, llvm::ArrayRef<uint64_t>(parts)), expected
    );
    EXPECT_EQ(
        FieldSpecAttr::getChecked(
            loc, &ctx, llvm::StringRef("custom"), llvm::ArrayRef<uint64_t>(parts)
        ),
        expected
    );
  }
  EXPECT_EQ(diagnostics, 12U);
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
