//===-- FieldTests.cpp - Unit tests for field methods -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestUtils.h"

#include "llzk/Util/Field.h"

#include <gtest/gtest.h>
#include <string>

using namespace llvm;
using namespace llzk;

static const Field &f = Field::getField("babybear");

struct FieldTests : public testing::TestWithParam<DynamicAPInt> {
  static const std::vector<DynamicAPInt> &TestingValues() {
    static std::vector<DynamicAPInt> vals = {f.zero(), f.one(), f.half(), f.maxVal(), f.prime()};
    return vals;
  }
};

TEST_F(FieldTests, Negatives) {
  // -a == b mod p s.t. a + b mod p = 0
  // In other words, -a = p - a
  ASSERT_EQ(f.maxVal(), f.reduce(-f.one()));
  ASSERT_EQ(f.zero(), f.reduce(f.felt(7) - f.felt(7)));
}

TEST_F(FieldTests, ToSigned) {
  DynamicAPInt p = f.prime();
  DynamicAPInt signedVal = f.toSigned(p);
  ASSERT_EQ(f.zero(), signedVal);
}

TEST(FieldAliasTests, BuiltinAliasesUseExpectedPrimes) {
  auto bn128 = Field::tryGetField("bn128");
  auto bn254 = Field::tryGetField("bn254");
  auto grumpkin = Field::tryGetField("grumpkin");
  ASSERT_TRUE(succeeded(bn128));
  ASSERT_TRUE(succeeded(bn254));
  ASSERT_TRUE(succeeded(grumpkin));
  EXPECT_EQ(
      bn128->get().prime(),
      toDynamicAPInt(
          "21888242871839275222246405745257275088548364400416034343698204186575808495617"
      )
  );
  EXPECT_EQ(bn128->get().prime(), bn254->get().prime());
  EXPECT_EQ(
      grumpkin->get().prime(),
      toDynamicAPInt(
          "21888242871839275222246405745257275088696311157297823662689037894645226208583"
      )
  );
}

//===------------------------------------------------------------------===//
// Suite of tests over all `TestingValues()`
//===------------------------------------------------------------------===//

TEST_P(FieldTests, DoubleNegatives) {
  auto p = f.reduce(GetParam());
  auto neg = f.reduce(-p);
  auto doubleNeg = f.reduce(-neg);
  ASSERT_EQ(p, doubleNeg);
}

TEST_P(FieldTests, ReducedToSignedInverses) {
  auto p = f.reduce(GetParam());
  auto signedVal = f.toSigned(p);
  auto reducedVal = f.reduce(signedVal);
  ASSERT_EQ(p, reducedVal);
}

INSTANTIATE_TEST_SUITE_P(FieldValSuite, FieldTests, testing::ValuesIn(FieldTests::TestingValues()));
