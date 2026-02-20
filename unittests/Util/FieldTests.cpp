//===-- FieldTests.cpp - Unit tests for field methods -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/Field.h"

#include <gtest/gtest.h>
#include <string>

#include "../LLZKTestUtils.h"

using namespace llvm;
using namespace llzk;

static const Field &f = Field::getField("babybear");

struct FieldTests : public testing::TestWithParam<DynamicAPInt> {
  static const std::vector<DynamicAPInt> &TestingValues() {
    static std::vector<DynamicAPInt> vals = {f.zero(), f.one(), f.half(), f.maxVal()};
    return vals;
  }
};

TEST_F(FieldTests, Negatives) {
  // -a == b mod p s.t. a + b mod p = 0
  // In other words, -a = p - a
  ASSERT_EQ(f.maxVal(), f.reduce(-f.one()));
  ASSERT_EQ(f.zero(), f.reduce(f.felt(7) - f.felt(7)));
}

TEST_P(FieldTests, DoubleNegatives) {
  auto p = f.reduce(GetParam());
  auto neg = f.reduce(-p);
  auto doubleNeg = f.reduce(-neg);
  ASSERT_EQ(p, doubleNeg);
}

INSTANTIATE_TEST_SUITE_P(FieldValSuite, FieldTests, testing::ValuesIn(FieldTests::TestingValues()));
