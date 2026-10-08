//===-- DynamicAPIntTests.cpp - Tests for DynamicAPInt helpers --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestUtils.h"

#include "llzk/Util/BinaryBuffer.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

using namespace llvm;
using namespace llzk;
using namespace std;

/// Return the Goldilocks prime, initialized when first needed by a test.
static const DynamicAPInt &getGoldilocks() {
  static const DynamicAPInt prime = toDynamicAPInt("18446744069414584321");
  return prime;
}

/// Return the BN254 prime, initialized when first needed by a test.
static const DynamicAPInt &getBN254() {
  static const DynamicAPInt prime = toDynamicAPInt(
      "21888242871839275222246405745257275088548364400416034343698204186575808495617"
  );
  return prime;
}

static void extendAPSInts(APSInt &a, APSInt &b) {
  unsigned maxBitwidth = max(a.getBitWidth(), b.getBitWidth());
  a = a.extend(maxBitwidth);
  b = b.extend(maxBitwidth);
}

//===----------------------------------------------------------------------===//
// Test conversions between DynamicAPInt and APSInt
//===----------------------------------------------------------------------===//

struct DynamicAPIntUnaryTest : public testing::TestWithParam<DynamicAPInt> {
  static const std::vector<DynamicAPInt> &TestingValues() {
    static std::vector<DynamicAPInt> vals = {
        DynamicAPInt(-1),
        DynamicAPInt(0),
        DynamicAPInt(1),
        DynamicAPInt(1234),
        DynamicAPInt(std::numeric_limits<int64_t>::min()),
        DynamicAPInt(std::numeric_limits<int64_t>::max()),
        DynamicAPInt(2013265921), // babybear
        DynamicAPInt(2147483647), // mersenne31
        DynamicAPInt(2130706433), // koalabear
        getGoldilocks(),
        -1 * getGoldilocks(),
        getBN254(),
        -1 * getBN254(),
    };
    return vals;
  }
};

TEST_P(DynamicAPIntUnaryTest, Conversions) {
  const DynamicAPInt &p = GetParam();
  DynamicAPInt convert = toDynamicAPInt(toAPSInt(p));
  ASSERT_EQ(p, convert);
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntUnaryTest, testing::ValuesIn(DynamicAPIntUnaryTest::TestingValues())
);

struct DynamicAPIntStringTest : public testing::TestWithParam<std::string> {
  static const std::vector<std::string> &TestingValues() {
    static std::vector<std::string> vals = {
        std::to_string(std::numeric_limits<int64_t>::min()),
        std::to_string(std::numeric_limits<int64_t>::max()),
        "2013265921",           // babybear
        "2147483647",           // mersenne31
        "2130706433",           // koalabear
        "18446744069414584321", // goldilocks
        "21888242871839275222246405745257275088548364400416034343698204186575808495617", // bn254
        "0",
    };
    return vals;
  }
};

TEST_P(DynamicAPIntStringTest, Strings) {
  const std::string &p = GetParam();
  APSInt input = APSInt(p);
  APSInt output = toAPSInt(toDynamicAPInt(p));
  // `toAPSInt()` always produces signed values so ensure signedness matches
  output.setIsUnsigned(input.isUnsigned());
  ASSERT_TRUE(APSInt::isSameValue(input, output));
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntStringTest, testing::ValuesIn(DynamicAPIntStringTest::TestingValues())
);

// Verify that size_t values above INT64_MAX are not mis-converted to negative.
// SIZE_MAX has its MSB set; if the APSInt wrapper incorrectly interpreted the value
// as signed, it would yield -1 instead of 18446744073709551615 (on 64-bit systems).
TEST(DynamicAPIntSizeTTest, SizeMax1) {
  DynamicAPInt a = toDynamicAPInt(SIZE_MAX);

  std::string buffer;
  llvm::raw_string_ostream(buffer) << a;

  ASSERT_EQ(buffer, std::to_string(SIZE_MAX));
}

TEST(DynamicAPIntSizeTTest, SizeMax2) {
  APSInt a = toAPSInt(toDynamicAPInt(SIZE_MAX));

  std::string buffer;
  llvm::raw_string_ostream(buffer) << a;

  ASSERT_EQ(buffer, std::to_string(SIZE_MAX));
}

//===----------------------------------------------------------------------===//
// Test bitwise AND, OR, XOR operations
//===----------------------------------------------------------------------===//

struct DynamicAPIntBinaryTest
    : public testing::TestWithParam<std::pair<DynamicAPInt, DynamicAPInt>> {
  static const std::vector<std::pair<DynamicAPInt, DynamicAPInt>> &TestingValues() {
    static std::vector<std::pair<DynamicAPInt, DynamicAPInt>> vals = {
        {DynamicAPInt(-1), DynamicAPInt(0)},
        {DynamicAPInt(-3), DynamicAPInt(2)},
        {-getBN254(), DynamicAPInt(-3)},
        {DynamicAPInt(-1), getBN254()},
        {DynamicAPInt(0xcafe), DynamicAPInt(0xdeadbeef)}
    };
    return vals;
  }
};

TEST_P(DynamicAPIntBinaryTest, BitAnd) {
  auto [a, b] = GetParam();
  // Commutative
  ASSERT_EQ(a & b, b & a);
  // Equivalent to APSInt operator
  APSInt sa = toAPSInt(a), sb = toAPSInt(b);
  extendAPSInts(sa, sb);
  DynamicAPInt baseline = toDynamicAPInt(sa & sb);
  ASSERT_EQ(a & b, baseline);
}

TEST_P(DynamicAPIntBinaryTest, BitOr) {
  auto [a, b] = GetParam();
  // Commutative
  ASSERT_EQ(a | b, b | a);
  // Equivalent to APSInt operator
  APSInt sa = toAPSInt(a), sb = toAPSInt(b);
  extendAPSInts(sa, sb);
  ASSERT_EQ(a | b, toDynamicAPInt(sa | sb));
}

TEST_P(DynamicAPIntBinaryTest, BitXor) {
  auto [a, b] = GetParam();
  // Commutative
  ASSERT_EQ(a ^ b, b ^ a);
  // Equivalent to APSInt operator
  APSInt sa = toAPSInt(a), sb = toAPSInt(b);
  extendAPSInts(sa, sb);
  ASSERT_EQ(a ^ b, toDynamicAPInt(sa ^ sb));
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntBinaryTest, testing::ValuesIn(DynamicAPIntBinaryTest::TestingValues())
);

//===----------------------------------------------------------------------===//
// Test left and right shift operations
//===----------------------------------------------------------------------===//

struct DynamicAPIntShiftTest : public testing::TestWithParam<std::pair<DynamicAPInt, unsigned>> {
  static const std::vector<std::pair<DynamicAPInt, unsigned>> &TestingValues() {
    static std::vector<std::pair<DynamicAPInt, unsigned>> vals = {
        {DynamicAPInt(-1), 0}, {getBN254(), 0},         {getBN254(), 32},
        {getBN254(), 100},     {DynamicAPInt(100), 32}, {DynamicAPInt(100), 100},
    };
    return vals;
  }
};

TEST_P(DynamicAPIntShiftTest, ShiftLeft) {
  auto [a, b] = GetParam();
  // Equivalent to APSInt operator
  APSInt base = toAPSInt(a);
  base = base.extend(base.getBitWidth() + b);

  ASSERT_EQ(a << toDynamicAPInt(APSInt::get(b)), toDynamicAPInt(base << b));
}

TEST_P(DynamicAPIntShiftTest, ShiftRight) {
  auto [a, b] = GetParam();
  // Equivalent to APSInt operator
  APSInt base = toAPSInt(a);
  base = base.extend(max(base.getBitWidth(), b));

  ASSERT_EQ(a >> toDynamicAPInt(APSInt::get(b)), toDynamicAPInt(base >> b));
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntShiftTest, testing::ValuesIn(DynamicAPIntShiftTest::TestingValues())
);

TEST(DynamicAPIntSafetyTest, NumericHashIgnoresRepresentation) {
  for (int64_t value : {-3, -1, 0, 1, 7}) {
    DynamicAPInt small(value);
    DynamicAPInt wide(APInt(256, value, true));
    DynamicAPInt computed = getBN254() + small - getBN254();
    EXPECT_EQ(small, wide);
    EXPECT_EQ(small, computed);
    EXPECT_EQ(hashDynamicAPInt(small), hashDynamicAPInt(wide));
    EXPECT_EQ(hashDynamicAPInt(small), hashDynamicAPInt(computed));
  }
  APInt big = APInt::getOneBitSet(128, 100);
  EXPECT_EQ(hashDynamicAPInt(DynamicAPInt(big)), hashDynamicAPInt(DynamicAPInt(big.sext(512))));
}

TEST(DynamicAPIntSafetyTest, CheckedParsing) {
  for (StringRef text : {"", "-", "+1", " 1", "1x", "1.0", "0x10"}) {
    auto value = parseDynamicAPInt(text);
    EXPECT_FALSE(value);
    if (!value) {
      consumeError(value.takeError());
    }
  }
  for (StringRef text : {"0", "-0", "0001", "-3", "18446744073709551616"}) {
    auto value = parseDynamicAPInt(text);
    ASSERT_TRUE(value);
    EXPECT_EQ(*value, toDynamicAPInt(text));
  }
}

TEST(DynamicAPIntSafetyTest, CheckedNarrowing) {
  auto signedMax = checkedToInt64(DynamicAPInt(INT64_MAX));
  ASSERT_TRUE(signedMax);
  EXPECT_EQ(*signedMax, INT64_MAX);
  auto unsignedMax = checkedToUInt64(toDynamicAPInt(SIZE_MAX));
  ASSERT_TRUE(unsignedMax);
  EXPECT_EQ(*unsignedMax, SIZE_MAX);
  auto signedMin = checkedToInt64(DynamicAPInt(INT64_MIN));
  ASSERT_TRUE(signedMin);
  EXPECT_EQ(*signedMin, INT64_MIN);
  auto tooBig = checkedToInt64(DynamicAPInt(INT64_MAX) + 1);
  EXPECT_FALSE(tooBig);
  consumeError(tooBig.takeError());
  auto tooSmall = checkedToInt64(DynamicAPInt(INT64_MIN) - 1);
  EXPECT_FALSE(tooSmall);
  consumeError(tooSmall.takeError());
  for (int64_t value : {INT64_MIN, int64_t(-1), int64_t(0), INT64_MAX}) {
    auto wide = checkedToInt64(DynamicAPInt(APInt(256, value, true)));
    ASSERT_TRUE(wide);
    EXPECT_EQ(*wide, value);
  }
  auto negative = checkedToUInt64(DynamicAPInt(-1));
  EXPECT_FALSE(negative);
  consumeError(negative.takeError());
  auto unsignedOverflow = checkedToUInt64(toDynamicAPInt(SIZE_MAX) + 1);
  EXPECT_FALSE(unsignedOverflow);
  consumeError(unsignedOverflow.takeError());
  auto zeroWidth = checkedToAPInt(DynamicAPInt(0), 0, true);
  EXPECT_FALSE(zeroWidth);
  consumeError(zeroWidth.takeError());
  auto signedOverflow = checkedToAPInt(DynamicAPInt(128), 8, true);
  EXPECT_FALSE(signedOverflow);
  consumeError(signedOverflow.takeError());
  auto byte = checkedToAPInt(DynamicAPInt(255), 8, false);
  ASSERT_TRUE(byte);
  EXPECT_EQ(byte->getBitWidth(), 8U);
  EXPECT_EQ(byte->getZExtValue(), 255U);
}

TEST(DynamicAPIntSafetyTest, HugeRightShiftAndSignedLeftShift) {
  auto huge = toDynamicAPInt("184467440737095516160000");
  EXPECT_EQ(DynamicAPInt(42) >> huge, DynamicAPInt(0));
  EXPECT_EQ(DynamicAPInt(-42) >> huge, DynamicAPInt(-1));
  EXPECT_EQ(DynamicAPInt(-3) << DynamicAPInt(100), -3 * (DynamicAPInt(1) << DynamicAPInt(100)));
  EXPECT_EQ(DynamicAPInt(-3) >> DynamicAPInt(1), DynamicAPInt(-2));
  auto unsignedMax = toDynamicAPInt(APSInt::getUnsigned(std::numeric_limits<unsigned>::max()));
  EXPECT_EQ(DynamicAPInt(42) >> unsignedMax, DynamicAPInt(0));
  EXPECT_EQ(DynamicAPInt(-42) >> unsignedMax, DynamicAPInt(-1));
  EXPECT_DEATH(DynamicAPInt(1) << unsignedMax, "invalid or unrepresentable left shift");
  EXPECT_EQ(DynamicAPInt(0) << (unsignedMax - 1), DynamicAPInt(0));
  DynamicAPInt wideCount(APInt(256, 1));
  EXPECT_EQ(DynamicAPInt(3) << wideCount, DynamicAPInt(6));
  EXPECT_EQ(DynamicAPInt(-3) >> wideCount, DynamicAPInt(-2));
}

TEST(DynamicAPIntSafetyTest, ModularExponentiationNormalizesInputs) {
  EXPECT_EQ(modExp(DynamicAPInt(-3), DynamicAPInt(3), DynamicAPInt(17)), DynamicAPInt(7));
  EXPECT_EQ(modExp(DynamicAPInt(3), DynamicAPInt(0), DynamicAPInt(1)), DynamicAPInt(0));
}

// Negative representatives must pass the inverse postcondition in assertion-enabled builds.
TEST(DynamicAPIntSafetyTest, ModularInverseAcceptsNegativeRepresentatives) {
  EXPECT_EQ(modInversePrime(DynamicAPInt(-3), DynamicAPInt(17)), DynamicAPInt(11));
  EXPECT_EQ(modInversePrime(DynamicAPInt(-20), DynamicAPInt(17)), DynamicAPInt(11));
}

TEST(DynamicAPIntSafetyTest, FieldEncodingRejectsInvalidValuesWithoutAppending) {
  BinaryBuffer buffer;
  ASSERT_FALSE(buffer.writeFieldElement(2, DynamicAPInt(0x1234)));
  EXPECT_EQ(buffer.bytes(), (llvm::ArrayRef<char> {'\x34', '\x12'}));
  for (auto [size, value] : std::vector<std::pair<uint32_t, DynamicAPInt>> {
           {0, DynamicAPInt(0)},
           {1, DynamicAPInt(-1)},
           {1, DynamicAPInt(256)},
           {UINT32_MAX, DynamicAPInt(0)}
       }) {
    auto error = buffer.writeFieldElement(size, value);
    EXPECT_TRUE(static_cast<bool>(error));
    consumeError(std::move(error));
    EXPECT_EQ(buffer.size(), 2U);
  }
}
