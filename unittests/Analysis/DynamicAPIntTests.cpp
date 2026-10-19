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
#include <tuple>

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

/// Check signed values at the native-integer conversion boundary, independent of width.
struct DynamicAPIntSignedConversionTest : public testing::TestWithParam<std::string> {};

TEST_P(DynamicAPIntSignedConversionTest, APSIntConversionPreservesSignedValues) {
  APSInt expected(APInt(128, GetParam(), 10), false);
  APSInt actual = toAPSInt(DynamicAPInt(expected));
  EXPECT_TRUE(actual.isSigned());
  EXPECT_TRUE(APSInt::isSameValue(actual, expected));
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntSignedConversionTest,
    testing::Values(
        "0", "-1", "9223372036854775807", "9223372036854775808", "-9223372036854775808",
        "-9223372036854775809", "18446744073709551615", "18446744073709551616",
        "-18446744073709551616", "-18446744073709551617"
    ),
    [](const testing::TestParamInfo<DynamicAPIntSignedConversionTest::ParamType> &paramInfo) {
      const std::string &text = paramInfo.param;
      return text.front() == '-' ? "Negative" + text.substr(1) : "NonNegative" + text;
    }
);

/// Check both signs around powers of two without retaining redundant storage words.
struct DynamicAPIntWordBoundaryTest
    : public testing::TestWithParam<std::tuple<unsigned, int, bool>> {};

TEST_P(DynamicAPIntWordBoundaryTest, APSIntConversionAtWordBoundaries) {
  auto [exponent, offset, negative] = GetParam();
  APInt power = APInt::getOneBitSet(exponent + 3, exponent);
  APInt magnitude = power + APInt(power.getBitWidth(), offset, /*isSigned=*/true);
  APSInt expected(negative ? -magnitude : magnitude, false);
  APSInt actual = toAPSInt(DynamicAPInt(expected));
  EXPECT_TRUE(actual.isSigned());
  EXPECT_TRUE(APSInt::isSameValue(actual, expected));
  EXPECT_EQ(actual.getNumWords(), APInt::getNumWords(expected.getSignificantBits()));
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntWordBoundaryTest,
    testing::Combine(
        testing::Values(
            63, 64, 65, 127, 128, 129, 191, 192, 193, 253, 254, 255, 256, 257, 319, 320, 321, 1023,
            1024, 1025
        ),
        testing::Values(-1, 0, 1), testing::Bool()
    ),
    ([](const testing::TestParamInfo<DynamicAPIntWordBoundaryTest::ParamType> &paramInfo) {
      auto [exponent, offset, negative] = paramInfo.param;
      const char *offsetName = offset < 0 ? "MinusOne" : (offset > 0 ? "PlusOne" : "Zero");
      return "Exponent" + std::to_string(exponent) + "Offset" + offsetName +
             (negative ? "Negative" : "Positive");
    })
);

/// Check both signs across the field-size and inline decimal-buffer boundaries.
struct DynamicAPIntDecimalBufferBoundaryTest
    : public testing::TestWithParam<std::tuple<unsigned, bool>> {};

TEST_P(DynamicAPIntDecimalBufferBoundaryTest, APSIntConversionAcrossDecimalBufferBoundaries) {
  auto [digits, negative] = GetParam();
  std::string text = (negative ? "-" : "") + std::string(digits, '9');
  APSInt expected(APInt(digits * 4 + 1, text, 10), false);
  APSInt actual = toAPSInt(DynamicAPInt(expected));
  EXPECT_TRUE(actual.isSigned());
  EXPECT_TRUE(APSInt::isSameValue(actual, expected));
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntDecimalBufferBoundaryTest,
    testing::Combine(testing::Values(77, 78, 95, 96, 97), testing::Bool()),
    ([](const testing::TestParamInfo<DynamicAPIntDecimalBufferBoundaryTest::ParamType> &paramInfo) {
      auto [digits, negative] = paramInfo.param;
      return "Digits" + std::to_string(digits) + (negative ? "Negative" : "Positive");
    })
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
        {DynamicAPInt(-1), 0},   {getBN254(), 0},
        {getBN254(), 32},        {getBN254(), 100},
        {DynamicAPInt(100), 32}, {DynamicAPInt(100), 100},
        {-getBN254(), 32},       {DynamicAPInt(std::numeric_limits<int64_t>::min()), 64},
        {DynamicAPInt(-3), 1},   {DynamicAPInt(APInt(256, 3)), 1},
    };
    return vals;
  }
};

TEST_P(DynamicAPIntShiftTest, ShiftLeft) {
  auto [a, b] = GetParam();
  // Equivalent to APSInt operator
  APSInt base = toAPSInt(a);
  base = base.extend(base.getBitWidth() + b);

  DynamicAPInt expected = toDynamicAPInt(base << b);
  ASSERT_EQ(a << b, expected);
  ASSERT_EQ(a << DynamicAPInt(b), expected);
}

TEST_P(DynamicAPIntShiftTest, ShiftRight) {
  auto [a, b] = GetParam();
  // Equivalent to APSInt operator
  APSInt base = toAPSInt(a);
  base = base.extend(max(base.getBitWidth(), b));

  DynamicAPInt expected = toDynamicAPInt(base >> b);
  ASSERT_EQ(a >> b, expected);
  ASSERT_EQ(a >> DynamicAPInt(b), expected);
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

/// Cover both native extraction paths, including values with redundant sign bits.
struct DynamicAPIntCheckedToSizeBoundaryTest : public testing::TestWithParam<size_t> {
  /// Return values around the signed and unsigned native-integer boundaries.
  static const std::vector<size_t> &TestingValues() {
    constexpr size_t sizeMax = std::numeric_limits<size_t>::max();
    static const std::vector<size_t> vals = {0,           1,      sizeMax / 2, sizeMax / 2 + 1,
                                             sizeMax - 1, sizeMax};
    return vals;
  }
};

TEST_P(DynamicAPIntCheckedToSizeBoundaryTest, NativeRepresentation) {
  size_t value = GetParam();
  auto checked = checkedToSize(llzk::toDynamicAPInt(value), "unit-test boundary");
  ASSERT_TRUE(static_cast<bool>(checked)) << llvm::toString(checked.takeError());
  EXPECT_EQ(*checked, value);
}

TEST_P(DynamicAPIntCheckedToSizeBoundaryTest, RedundantSignBits) {
  size_t value = GetParam();
  auto checked = checkedToSize(llvm::DynamicAPInt(llvm::APInt(256, value)), "unit-test boundary");
  ASSERT_TRUE(static_cast<bool>(checked)) << llvm::toString(checked.takeError());
  EXPECT_EQ(*checked, value);
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntCheckedToSizeBoundaryTest,
    testing::ValuesIn(DynamicAPIntCheckedToSizeBoundaryTest::TestingValues()),
    testing::PrintToStringParamName()
);

/// Preserve overflow diagnostics for near-boundary and arbitrarily large values.
struct DynamicAPIntCheckedToSizeOverflowTest : public testing::TestWithParam<DynamicAPInt> {};

TEST_P(DynamicAPIntCheckedToSizeOverflowTest, RejectsOverflow) {
  auto checked = checkedToSize(GetParam(), "unit-test overflow");
  ASSERT_FALSE(static_cast<bool>(checked));
  EXPECT_EQ(llvm::toString(checked.takeError()), "unit-test overflow would overflow size_t");
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntCheckedToSizeOverflowTest,
    testing::Values(
        // Leave a zero sign bit above SIZE_MAX + 1.
        llvm::DynamicAPInt(
            llvm::APInt::getOneBitSet(
                sizeof(size_t) * CHAR_BIT + 2, std::numeric_limits<size_t>::digits
            )
        ),
        llvm::DynamicAPInt(llvm::APInt::getOneBitSet(256, 200))
    ),
    testing::PrintToStringParamName()
);

/// Negative values retain the underflow diagnostic regardless of their magnitude.
struct DynamicAPIntCheckedToSizeUnderflowTest : public testing::TestWithParam<DynamicAPInt> {};

TEST_P(DynamicAPIntCheckedToSizeUnderflowTest, RejectsUnderflow) {
  auto checked = checkedToSize(GetParam(), llvm::Twine("unit-test") + " underflow");
  ASSERT_FALSE(static_cast<bool>(checked));
  EXPECT_EQ(llvm::toString(checked.takeError()), "unit-test underflow would underflow size_t");
}

INSTANTIATE_TEST_SUITE_P(
    , DynamicAPIntCheckedToSizeUnderflowTest,
    testing::Values(
        llvm::DynamicAPInt(-1), -llvm::DynamicAPInt(llvm::APInt::getOneBitSet(256, 200))
    ),
    testing::PrintToStringParamName()
);

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
  EXPECT_EQ(DynamicAPInt(-3) << 100, -3 * (DynamicAPInt(1) << 100));
  EXPECT_EQ(DynamicAPInt(-3) >> 1, DynamicAPInt(-2));
  const DynamicAPInt unsignedMax(std::numeric_limits<unsigned>::max());
  EXPECT_EQ(DynamicAPInt(42) >> unsignedMax, DynamicAPInt(0));
  EXPECT_EQ(DynamicAPInt(-42) >> unsignedMax, DynamicAPInt(-1));
  EXPECT_DEATH(DynamicAPInt(1) << unsignedMax, "invalid or unrepresentable left shift");
  EXPECT_EQ(DynamicAPInt(0) << (unsignedMax - 1), DynamicAPInt(0));
  DynamicAPInt wideCount(APInt(256, 1));
  EXPECT_EQ(DynamicAPInt(3) << wideCount, DynamicAPInt(6));
  EXPECT_EQ(DynamicAPInt(-3) >> wideCount, DynamicAPInt(-2));
}

/// Native counts preserve saturation and reject invalid shifts before narrowing or allocation.
TEST(DynamicAPIntSafetyTest, NativeShiftCountBoundaries) {
  for (int64_t count : std::vector<int64_t> {
           63, 64, 65, std::numeric_limits<unsigned>::max(), std::numeric_limits<int64_t>::max()
       }) {
    SCOPED_TRACE(count);
    EXPECT_EQ(DynamicAPInt(42) >> count, DynamicAPInt(0));
    EXPECT_EQ(DynamicAPInt(-42) >> count, DynamicAPInt(-1));
    EXPECT_EQ(DynamicAPInt(0) >> count, DynamicAPInt(0));
    EXPECT_EQ(DynamicAPInt(42) >> DynamicAPInt(count), DynamicAPInt(0));
    EXPECT_EQ(DynamicAPInt(-42) >> DynamicAPInt(count), DynamicAPInt(-1));
  }
  for (int64_t count : {int64_t(-1), std::numeric_limits<int64_t>::min()}) {
    EXPECT_DEATH(DynamicAPInt(1) << count, "invalid or unrepresentable left shift");
    EXPECT_DEATH(DynamicAPInt(1) >> count, "negative right shift");
    EXPECT_DEATH(DynamicAPInt(1) << DynamicAPInt(count), "invalid or unrepresentable left shift");
    EXPECT_DEATH(DynamicAPInt(1) >> DynamicAPInt(count), "negative right shift");
  }
  int64_t unsignedMax = std::numeric_limits<unsigned>::max();
  EXPECT_EQ(DynamicAPInt(0) << (unsignedMax - 1), DynamicAPInt(0));
  EXPECT_DEATH(DynamicAPInt(0) << unsignedMax, "invalid or unrepresentable left shift");
  EXPECT_DEATH(
      DynamicAPInt(1) << std::numeric_limits<int64_t>::max(),
      "invalid or unrepresentable left shift"
  );
}

/// Counts outside int64_t must be handled before delegating to the native overloads.
TEST(DynamicAPIntSafetyTest, ShiftCountOutsideInt64) {
  auto count = toDynamicAPInt("9223372036854775808");
  EXPECT_EQ(DynamicAPInt(42) >> count, DynamicAPInt(0));
  EXPECT_EQ(DynamicAPInt(-42) >> count, DynamicAPInt(-1));
  EXPECT_EQ(DynamicAPInt(0) >> count, DynamicAPInt(0));
  EXPECT_DEATH(DynamicAPInt(1) << count, "invalid or unrepresentable left shift");
  auto negativeCount = -count - 1;
  EXPECT_DEATH(DynamicAPInt(1) << negativeCount, "invalid or unrepresentable left shift");
  EXPECT_DEATH(DynamicAPInt(1) >> negativeCount, "negative right shift");
}

TEST(DynamicAPIntSafetyTest, ModularExponentiationNormalizesInputs) {
  EXPECT_EQ(modExp(DynamicAPInt(-3), DynamicAPInt(3), DynamicAPInt(17)), DynamicAPInt(7));
  EXPECT_EQ(modExp(DynamicAPInt(3), DynamicAPInt(0), DynamicAPInt(1)), DynamicAPInt(0));
}

/// Preserve exponent bits across compaction, including initially wide-stored small values.
TEST(DynamicAPIntSafetyTest, ModularExponentiationCompactsExponent) {
  // Powers of 3 modulo 17 repeat every 16 exponents.
  for (const auto &[text, expected] : std::vector<std::pair<StringRef, int64_t>> {
           {"0", 1},
           {"1", 3},
           {"2", 9},
           {"65537", 3},
           {"9223372036854775807", 6},
           {"9223372036854775808", 1},
           {"9223372036854775809", 3},
           {"18446744073709551615", 6},
           {"18446744073709551616", 1}
       }) {
    SCOPED_TRACE(text.str());
    for (const DynamicAPInt &exponent :
         {toDynamicAPInt(text), DynamicAPInt(APInt(256, text, 10))}) {
      EXPECT_EQ(modExp(DynamicAPInt(3), exponent, DynamicAPInt(17)), DynamicAPInt(expected));
    }
  }
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
  for (const auto &[size, value] : std::vector<std::pair<uint32_t, DynamicAPInt>> {
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
