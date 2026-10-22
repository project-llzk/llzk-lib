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
#include <unordered_set>

using namespace llvm;
using namespace llzk;

struct FieldTests : public testing::TestWithParam<DynamicAPInt> {
  const Field &f = Field::getField("babybear");

  static const std::vector<DynamicAPInt> &TestingValues() {
    const Field &field = Field::getField("babybear");
    static std::vector<DynamicAPInt> vals = {
        field.zero(), field.one(), field.half(), field.maxVal(), field.prime()
    };
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

// Hashing follows prime equality even for aliases and independently copied fields.
TEST(FieldAliasTests, HashUsesPrimeIdentity) {
  const Field &bn128 = Field::getField("bn128");
  const Field &bn254 = Field::getField("bn254");
  const Field &grumpkin = Field::getField("grumpkin");
  const Field copy = bn128;
  const Field::Hash hash;

  EXPECT_EQ(bn128, bn254);
  EXPECT_EQ(bn128, copy);
  EXPECT_NE(bn128, grumpkin);
  EXPECT_EQ(hash(bn128), hash(bn254));
  EXPECT_EQ(hash(bn128), hash(copy));

  std::unordered_set<Field, Field::Hash> fields;
  EXPECT_TRUE(fields.insert(bn128).second);
  EXPECT_FALSE(fields.insert(bn254).second);
  EXPECT_FALSE(fields.insert(copy).second);
  EXPECT_TRUE(fields.insert(grumpkin).second);
  EXPECT_EQ(fields.size(), 2U);
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

TEST(FieldBoundaryTests, InvalidModuliReportErrorsWithoutRegistration) {
  // Exercise both public overloads directly, bypassing FieldSpecAttr validation.
  mlir::MLIRContext ctx;
  for (int64_t prime : {-7, 0, 1}) {
    SCOPED_TRACE(prime);
    for (bool useString : {false, true}) {
      SCOPED_TRACE(useString ? "string overload" : "DynamicAPInt overload");
      StringRef name = useString ? "field-test-invalid-string" : "field-test-invalid-integer";
      ASSERT_TRUE(failed(Field::tryGetField(name)));
      unsigned callbacks = 0;
      std::vector<std::string> diagnostics;
      mlir::ScopedDiagnosticHandler handler(&ctx, [&](mlir::Diagnostic &diag) {
        diagnostics.push_back(diag.str());
        return mlir::success();
      });
      auto errFn = [&]() {
        ++callbacks;
        return InFlightDiagnosticWrapper(mlir::emitError(mlir::UnknownLoc::get(&ctx)));
      };
      if (useString) {
        Field::addField(name, std::to_string(prime), errFn);
      } else {
        Field::addField(name, DynamicAPInt(prime), errFn);
      }
      EXPECT_EQ(callbacks, 1U);
      EXPECT_EQ(diagnostics, (std::vector<std::string> {"field modulus must be at least 2"}));
      EXPECT_TRUE(failed(Field::tryGetField(name)));
    }
  }
}

TEST(FieldBoundaryTests, TwoElementFieldEncodesItsModulus) {
  Field::addField("dynamic-apint-test-two", DynamicAPInt(2), nullptr);
  const auto &field = Field::getField("dynamic-apint-test-two");
  EXPECT_EQ(field.bitWidth(), 2U);
  EXPECT_EQ(field.half(), DynamicAPInt(1));
  EXPECT_EQ(field.toSigned(DynamicAPInt(1)), DynamicAPInt(-1));
  EXPECT_EQ(field.reduce(DynamicAPInt(-1)), DynamicAPInt(1));
}
