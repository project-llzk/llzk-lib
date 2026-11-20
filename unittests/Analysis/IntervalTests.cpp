//===-- IntervalTests.cpp - Unit tests for interval analysis ----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"
#include "../LLZKTestUtils.h"
#include "IntervalTestUtils.h"

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Analysis/Intervals.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Parser/Parser.h>

#include <array>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

class IntervalTests : public testing::Test {
protected:
  const Field &f;
  const Interval empty, entire;

  IntervalTests()
      : f(Field::getField("babybear")), empty(Interval::Empty(f)), entire(Interval::Entire(f)) {}

  Interval degen(int64_t i) const { return Interval::Degenerate(f, DynamicAPInt(i)); }

  Interval interval(int64_t a, int64_t b) const { return UnreducedInterval(a, b).reduce(f); }

  inline static void
  AssertUnreducedIntervalEq(const UnreducedInterval &expected, const UnreducedInterval &actual) {
    ASSERT_TRUE(checkCond(expected, actual, expected == actual));
  }

  inline static void AssertIntervalEq(const Interval &expected, const Interval &actual) {
    ASSERT_TRUE(checkCond(expected, actual, expected == actual));
  }
};

// Field aliases share interval identity, while different primes remain distinct.
TEST_F(IntervalTests, EqualityAndHashIncludeFieldPrime) {
  auto intervals = [](const Field &field) {
    return std::array {
        Interval::Empty(field), Interval::Entire(field), Interval::Degenerate(field, field.one()),
        UnreducedInterval(1, 7).reduce(field)
    };
  };
  const auto bn128 = intervals(Field::getField("bn128"));
  const auto bn254 = intervals(Field::getField("bn254"));
  const auto grumpkin = intervals(Field::getField("grumpkin"));
  const Interval::Hash hash;

  for (size_t idx = 0; idx < bn128.size(); ++idx) {
    SCOPED_TRACE(idx);
    EXPECT_EQ(bn128[idx], bn254[idx]);
    EXPECT_EQ(hash(bn128[idx]), hash(bn254[idx]));
    EXPECT_NE(bn128[idx], grumpkin[idx]);

    std::unordered_set<Interval, Interval::Hash> values;
    EXPECT_TRUE(values.insert(bn128[idx]).second);
    EXPECT_FALSE(values.insert(bn254[idx]).second);
    EXPECT_TRUE(values.insert(grumpkin[idx]).second);
    EXPECT_EQ(values.size(), 2U);
  }
}

TEST_F(IntervalTests, UnreducedIntervalOverlap) {
  UnreducedInterval a(0, 100), b(100, 200), c(101, 300), d(1, 0);
  ASSERT_TRUE(a.overlaps(b));
  ASSERT_TRUE(b.overlaps(a));
  ASSERT_FALSE(a.overlaps(c));
  ASSERT_TRUE(b.overlaps(c));
  ASSERT_FALSE(d.overlaps(a));
}

TEST_F(IntervalTests, UnreducedIntervalWidth) {
  // Standard width.
  UnreducedInterval a(0, 100);
  ASSERT_EQ(f.felt(101), a.width());
  // Standard width for a single element range.
  UnreducedInterval b(4, 4);
  ASSERT_EQ(f.one(), b.width());
  // Range of this will be 0 since a > b.
  UnreducedInterval c(4, 3);
  ASSERT_EQ(f.zero(), c.width());
}

TEST_F(IntervalTests, IntervalWidth) {
  // Standard width.
  Interval a = UnreducedInterval(0, 100).reduce(f);
  ASSERT_EQ(f.felt(101), a.width());
  // Standard width for a single element range.
  Interval b = UnreducedInterval(4, 4).reduce(f);
  ASSERT_EQ(f.one(), b.width());
  // Range of this will be 0 since a > b.
  Interval c = UnreducedInterval(4, 3).reduce(f);
  ASSERT_EQ(f.zero(), c.width());

  ASSERT_EQ(Interval::Entire(f).width(), f.prime());
  ASSERT_EQ(Interval::Empty(f).width(), f.zero());
  ASSERT_EQ(Interval::Degenerate(f, f.felt(7)).width(), f.one());
}

TEST_F(IntervalTests, Partitions) {
  UnreducedInterval a(0, 100), b(100, 200), c(101, 300), d(1, 0), s1(1, 10), s2(3, 7);

  // Some basic overlapping intervals
  AssertUnreducedIntervalEq(a, a.computeLTPart(b));
  AssertUnreducedIntervalEq(a, a.computeLEPart(b));
  AssertUnreducedIntervalEq(b, b.computeGEPart(a));
  AssertUnreducedIntervalEq(b, b.computeGTPart(a));

  AssertUnreducedIntervalEq(UnreducedInterval(1, 6), s1.computeLTPart(s2));
  AssertUnreducedIntervalEq(UnreducedInterval(1, 7), s1.computeLEPart(s2));
  AssertUnreducedIntervalEq(UnreducedInterval(4, 10), s1.computeGTPart(s2));
  AssertUnreducedIntervalEq(UnreducedInterval(3, 10), s1.computeGEPart(s2));

  // Some non-overlapping intervals, should all be empty
  ASSERT_TRUE(b.computeLTPart(a).reduce(f).isEmpty());
  ASSERT_TRUE(a.computeGTPart(b).reduce(f).isEmpty());
  ASSERT_TRUE(c.computeLEPart(a).reduce(f).isEmpty());
  ASSERT_TRUE(a.computeGEPart(c).reduce(f).isEmpty());

  // Any computation where LHS or RHS is empty returns LHS.
  AssertUnreducedIntervalEq(a, a.computeLTPart(d));
  AssertUnreducedIntervalEq(b, b.computeLEPart(d));
  AssertUnreducedIntervalEq(c, c.computeGTPart(d));
  AssertUnreducedIntervalEq(d, d.computeGEPart(d));
  AssertUnreducedIntervalEq(d, d.computeLTPart(a));
  AssertUnreducedIntervalEq(d, d.computeLEPart(b));
  AssertUnreducedIntervalEq(d, d.computeGTPart(c));
  AssertUnreducedIntervalEq(d, d.computeGEPart(d));
}

TEST_F(IntervalTests, Difference) {
  // Following the examples in the Interval::difference docs.
  auto a = Interval::TypeA(f, f.felt(1), f.felt(10));
  auto b = Interval::TypeA(f, f.felt(5), f.felt(11));
  auto c = Interval::TypeA(f, f.felt(5), f.felt(6));

  ASSERT_EQ(Interval::TypeA(f, f.felt(1), f.felt(4)), a.difference(b));
  ASSERT_EQ(a, a.difference(c));
}

TEST_F(IntervalTests, UnreduceReduce) {
  // unreducing and reducing should not be destructive
  AssertIntervalEq(Interval::Entire(f), Interval::Entire(f).toUnreduced().reduce(f));
  AssertIntervalEq(Interval::Empty(f), Interval::Empty(f).toUnreduced().reduce(f));
  AssertIntervalEq(
      Interval::Degenerate(f, f.felt(8)), Interval::Degenerate(f, f.felt(8)).toUnreduced().reduce(f)
  );
}

TEST_F(IntervalTests, AdditiveIdentities) {
  // Empty + Empty = Empty
  AssertIntervalEq(empty, empty + empty);
  // Entire + Entire = Entire
  AssertIntervalEq(entire, entire + entire);
  // Entire + Empty = Entire
  AssertIntervalEq(entire, entire + empty);
  AssertIntervalEq(entire, empty + entire);
}

TEST_F(IntervalTests, NegativeIdentities) {
  // negative "entire" should still be "entire"
  AssertIntervalEq(Interval::Entire(f), -Interval::Entire(f));

  // negative "empty" should still be "empty"
  AssertIntervalEq(Interval::Empty(f), -Interval::Empty(f));

  // -1 should be max value when reduced (1 + (-1) % p == 1 + (p - 1) % p == p % p == 0)
  auto maxValDegen = Interval::Degenerate(f, f.maxVal());
  auto oneDegen = Interval::Degenerate(f, f.one());
  AssertIntervalEq(maxValDegen, -oneDegen);
}

TEST_F(IntervalTests, BitwiseNot) {
  auto one = Interval::Degenerate(f, f.one());
  auto a = Interval::TypeA(f, f.zero(), f.felt(7));
  auto notA = Interval::TypeF(f, f.prime() - f.felt(6), f.one());
  AssertIntervalEq(~a, one - a);
  AssertIntervalEq(notA, ~a);
}

TEST_F(IntervalTests, BitwiseOr) {
  auto zero = Interval::Degenerate(f, f.zero());
  auto one = Interval::Degenerate(f, f.one());
  auto two = Interval::Degenerate(f, f.felt(2));
  auto oneToTwo = Interval::TypeA(f, f.one(), f.felt(2));

  AssertIntervalEq(Interval::Degenerate(f, f.felt(3)), one | two);
  AssertIntervalEq(oneToTwo, oneToTwo | zero);
  AssertIntervalEq(Interval::Entire(f), oneToTwo | one);
}

TEST_F(IntervalTests, BitwiseXor) {
  auto zero = Interval::Degenerate(f, f.zero());
  auto one = Interval::Degenerate(f, f.one());
  auto two = Interval::Degenerate(f, f.felt(2));
  auto oneToTwo = Interval::TypeA(f, f.one(), f.felt(2));

  AssertIntervalEq(Interval::Degenerate(f, f.felt(3)), one ^ two);
  AssertIntervalEq(oneToTwo, oneToTwo ^ zero);
  AssertIntervalEq(Interval::Entire(f), oneToTwo ^ one);
}

TEST_F(IntervalTests, BoolXor) {
  auto falseInterval = Interval::False(f);
  auto trueInterval = Interval::True(f);
  auto boolInterval = Interval::Boolean(f);

  AssertIntervalEq(falseInterval, boolXor(trueInterval, trueInterval));
  AssertIntervalEq(trueInterval, boolXor(trueInterval, falseInterval));
  AssertIntervalEq(boolInterval, boolXor(boolInterval, trueInterval));
}

TEST_F(IntervalTests, UnsignedIntDiv) {
  auto rangeTenToFifteen = Interval::TypeA(f, f.felt(10), f.felt(15));
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto five = Interval::Degenerate(f, f.felt(5));

  auto res0 = unsignedIntDiv(rangeTenToFifteen, five);
  ASSERT_TRUE(succeeded(res0));
  AssertIntervalEq(Interval::TypeA(f, f.felt(2), f.felt(3)), *res0);

  auto res1 = unsignedIntDiv(ten, rangeTenToFifteen);
  ASSERT_TRUE(succeeded(res1));
  AssertIntervalEq(Interval::TypeA(f, f.zero(), f.one()), *res1);
}

TEST_F(IntervalTests, UnsignedIntDivByZero) {
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto zeroToOne = Interval::TypeA(f, f.zero(), f.one());

  auto res = unsignedIntDiv(ten, zeroToOne);
  ASSERT_TRUE(failed(res));
}

TEST_F(IntervalTests, FeltDiv) {
  auto one = Interval::Degenerate(f, f.one());
  auto two = Interval::Degenerate(f, f.felt(2));
  auto invTwo = Interval::Degenerate(f, f.inv(f.felt(2)));

  auto res0 = feltDiv(one, two);
  ASSERT_TRUE(succeeded(res0));
  AssertIntervalEq(invTwo, *res0);
}

TEST_F(IntervalTests, FeltDivIntervalDivisorUnsupported) {
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto oneToTwo = Interval::TypeA(f, f.one(), f.felt(2));

  auto res = feltDiv(ten, oneToTwo);
  ASSERT_TRUE(failed(res));
}

TEST_F(IntervalTests, FeltDivByZero) {
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto zeroToOne = Interval::TypeA(f, f.zero(), f.one());

  auto res = feltDiv(ten, zeroToOne);
  ASSERT_TRUE(failed(res));
}

TEST_F(IntervalTests, SignedIntDiv) {
  auto rangeTenToFifteen = Interval::TypeA(f, f.felt(10), f.felt(15));
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto negTen = Interval::Degenerate(f, f.reduce(-10));
  auto negFifteenToTen = UnreducedInterval(-15, -10).reduce(f);

  auto res0 = signedIntDiv(ten, rangeTenToFifteen);
  ASSERT_TRUE(succeeded(res0));
  AssertIntervalEq(Interval::TypeA(f, f.zero(), f.one()), *res0);

  auto res1 = signedIntDiv(negTen, rangeTenToFifteen);
  ASSERT_TRUE(succeeded(res1));
  AssertIntervalEq(UnreducedInterval(-1, 0).reduce(f), *res1);

  auto res2 = signedIntDiv(negTen, negFifteenToTen);
  ASSERT_TRUE(succeeded(res2));
  AssertIntervalEq(Interval::TypeA(f, f.zero(), f.one()), *res2);

  auto res3 = signedIntDiv(negFifteenToTen, ten);
  ASSERT_TRUE(succeeded(res3));
  AssertIntervalEq(Interval::Degenerate(f, f.reduce(-1)), *res3);
}

TEST_F(IntervalTests, SignedIntDivByZero) {
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto minusOneToOne = UnreducedInterval(-1, 1).reduce(f);

  auto res = signedIntDiv(ten, minusOneToOne);
  ASSERT_TRUE(failed(res));
}

TEST_F(IntervalTests, SignedMod) {
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto negTen = Interval::Degenerate(f, f.reduce(-10));
  auto negSeven = Interval::Degenerate(f, f.reduce(-7));
  auto two = Interval::Degenerate(f, f.felt(2));
  auto rangeTenToFifteen = Interval::TypeA(f, f.felt(10), f.felt(15));

  AssertIntervalEq(Interval::Degenerate(f, f.zero()), signedMod(ten, two));
  AssertIntervalEq(Interval::Degenerate(f, f.reduce(-1)), signedMod(negSeven, two));
  AssertIntervalEq(Interval::TypeA(f, f.zero(), f.felt(10)), signedMod(ten, rangeTenToFifteen));
  AssertIntervalEq(UnreducedInterval(-10, 0).reduce(f), signedMod(negTen, rangeTenToFifteen));
}

TEST_F(IntervalTests, SignedModByZero) {
  auto ten = Interval::Degenerate(f, f.felt(10));
  auto minusOneToOne = UnreducedInterval(-1, 1).reduce(f);

  AssertIntervalEq(entire, signedMod(ten, minusOneToOne));
}

TEST_F(IntervalTests, Mod) {
  AssertIntervalEq(interval(0, 7), entire % degen(8));
  AssertIntervalEq(interval(0, 9), interval(0, 100) % interval(1, 10));
  AssertIntervalEq(entire, degen(7) % interval(0, 1000));
  AssertIntervalEq(empty, empty % empty);
  AssertIntervalEq(empty, entire % empty);
  AssertIntervalEq(empty, empty % entire);
  AssertIntervalEq(entire, degen(1) % entire);
  // any % typeF == entire
  auto typeF = UnreducedInterval(f.half() + f.one(), f.prime() + f.one()).reduce(f);
  ASSERT_TRUE(typeF.isTypeF());
  AssertIntervalEq(entire, interval(7, 8) % typeF);
}

class IntervalAnalysisAPITests : public LLZKTest {
protected:
  inline static void
  AssertUnreducedIntervalEq(const UnreducedInterval &expected, const UnreducedInterval &actual) {
    ASSERT_TRUE(checkCond(expected, actual, expected == actual));
  }

  static constexpr auto kArrayIntervalModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @ArrayIntervals {
    struct.member @out : !array.type<3 x !felt.type> {llzk.pub, signal}

    function.def @compute() -> !struct.type<@ArrayIntervals> attributes {function.allow_witness} {
      %self = struct.new : <@ArrayIntervals>
      function.return %self : !struct.type<@ArrayIntervals>
    }

    function.def @constrain(%arg0: !struct.type<@ArrayIntervals>) attributes {function.allow_constraint} {
      %0 = struct.readm %arg0[@out] : <@ArrayIntervals>, !array.type<3 x !felt.type>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %1 = array.read %0[%c0] : <3 x !felt.type>, !felt.type
      %2 = array.read %0[%c1] : <3 x !felt.type>, !felt.type
      %3 = array.read %0[%c2] : <3 x !felt.type>, !felt.type
      %felt_const_1 = felt.const  1
      %felt_const_2 = felt.const  2
      %felt_const_3 = felt.const  3
      constrain.eq %1, %felt_const_1 : !felt.type, !felt.type
      constrain.eq %2, %felt_const_2 : !felt.type, !felt.type
      constrain.eq %3, %felt_const_3 : !felt.type, !felt.type
      function.return
    }
  }
}
)mlir";

  static constexpr auto kComputeArrayMemberWriteModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @ComputeArrayMemberWrite {
    struct.member @out : !array.type<2 x !felt.type> {llzk.pub, signal}

    function.def @compute(%arg0: !felt.type) -> !struct.type<@ComputeArrayMemberWrite>
        attributes {function.allow_witness} {
      %self = struct.new : <@ComputeArrayMemberWrite>
      %felt_const_0 = felt.const  0
      %felt_const_5 = felt.const  5
      %felt_const_6 = felt.const  6
      %cmp0 = bool.cmp ge(%arg0, %felt_const_5) : !felt.type, !felt.type
      bool.assert %cmp0
      %cmp1 = bool.cmp le(%arg0, %felt_const_6) : !felt.type, !felt.type
      bool.assert %cmp1
      %0 = array.new %felt_const_0, %arg0 : <2 x !felt.type>
      struct.writem %self[@out] = %0 : <@ComputeArrayMemberWrite>, !array.type<2 x !felt.type>
      function.return %self : !struct.type<@ComputeArrayMemberWrite>
    }

    function.def @constrain(%arg0: !struct.type<@ComputeArrayMemberWrite>, %arg1: !felt.type)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedIntervalPropagationModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @TrackUnreduced {
    struct.member @out : !felt.type {llzk.pub, signal}

    function.def @compute(%a: !felt.type) -> !struct.type<@TrackUnreduced>
        attributes {function.allow_witness, function.allow_non_native_field_ops} {
      %self = struct.new : <@TrackUnreduced>
      %felt_const_5 = felt.const 5
      %sum = felt.add %a, %felt_const_5 : !felt.type, !felt.type
      %neg = felt.neg %sum : !felt.type
      %div = felt.uintdiv %sum, %felt_const_5
      struct.writem %self[@out] = %neg : <@TrackUnreduced>, !felt.type
      function.return %self : !struct.type<@TrackUnreduced>
    }

    function.def @constrain(%self: !struct.type<@TrackUnreduced>, %a: !felt.type)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedBoolAndSelectModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @TrackUnreducedBoolAndSelect {
    struct.member @out : !felt.type {llzk.pub, signal}

    function.def @compute(%flag: i1) -> !struct.type<@TrackUnreducedBoolAndSelect>
        attributes {function.allow_witness, function.allow_non_native_field_ops} {
      %self = struct.new : <@TrackUnreducedBoolAndSelect>
      %true = arith.constant true
      %false = arith.constant false
      %felt_const_3 = felt.const 3
      %felt_const_9 = felt.const 9
      %xor = bool.xor %true, %false
      %not = bool.not %flag
      %sel_true = arith.select %true, %felt_const_3, %felt_const_9 : !felt.type
      %sel_false = arith.select %false, %felt_const_3, %felt_const_9 : !felt.type
      %sel_flag = arith.select %flag, %felt_const_3, %felt_const_9 : !felt.type
      struct.writem %self[@out] = %sel_flag : <@TrackUnreducedBoolAndSelect>, !felt.type
      function.return %self : !struct.type<@TrackUnreducedBoolAndSelect>
    }

    function.def @constrain(%self: !struct.type<@TrackUnreducedBoolAndSelect>, %flag: i1)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedTypeFLiftModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @TrackUnreducedTypeF {
    struct.member @out : !felt.type {llzk.pub, signal}

    function.def @compute(%a: !felt.type) -> !struct.type<@TrackUnreducedTypeF>
        attributes {function.allow_witness, function.allow_non_native_field_ops} {
      %self = struct.new : <@TrackUnreducedTypeF>
      %felt_const_1 = felt.const 1
      %cmp = bool.cmp le(%a, %felt_const_1) : !felt.type, !felt.type
      bool.assert %cmp
      %neg = felt.neg %a : !felt.type
      struct.writem %self[@out] = %neg : <@TrackUnreducedTypeF>, !felt.type
      function.return %self : !struct.type<@TrackUnreducedTypeF>
    }

    function.def @constrain(%self: !struct.type<@TrackUnreducedTypeF>, %a: !felt.type)
        attributes {function.allow_constraint} {
      %read = struct.readm %self[@out] : <@TrackUnreducedTypeF>, !felt.type
      %felt_const_0 = felt.const 0
      %sum = felt.add %read, %felt_const_0 : !felt.type, !felt.type
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedTypeFPropagationModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @TrackUnreducedTypeFPropagation {
    function.def @compute(%a: !felt.type) -> !struct.type<@TrackUnreducedTypeFPropagation>
        attributes {function.allow_witness, function.allow_non_native_field_ops} {
      %self = struct.new : <@TrackUnreducedTypeFPropagation>
      function.return %self : !struct.type<@TrackUnreducedTypeFPropagation>
    }

    function.def @constrain(%self: !struct.type<@TrackUnreducedTypeFPropagation>, %a: !felt.type)
        attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      %felt_const_1 = felt.const 1
      %sum = felt.add %a, %felt_const_1 : !felt.type, !felt.type
      %cmp = bool.cmp le(%sum, %felt_const_1) : !felt.type, !felt.type
      bool.assert %cmp
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedSignalReadDefaultModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @TrackUnreducedSignalReadDefault {
    struct.member @sig : !felt.type {signal}

    function.def @compute() -> !struct.type<@TrackUnreducedSignalReadDefault>
        attributes {function.allow_witness} {
      %self = struct.new : <@TrackUnreducedSignalReadDefault>
      function.return %self : !struct.type<@TrackUnreducedSignalReadDefault>
    }

    function.def @constrain(%self: !struct.type<@TrackUnreducedSignalReadDefault>)
        attributes {function.allow_constraint} {
      %sig = struct.readm %self[@sig] : <@TrackUnreducedSignalReadDefault>, !felt.type
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedIsZeroModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @IsZero {
    struct.member @out : !felt.type {llzk.pub}
    struct.member @inv : !felt.type

    function.def @compute(%in: !felt.type) -> !struct.type<@IsZero>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@IsZero>
      %const_1 = felt.const 1
      %inv = felt.inv %in
      struct.writem %self[@inv] = %inv : <@IsZero>, !felt.type
      %neg_in = felt.neg %in : !felt.type
      %mul = felt.mul %neg_in, %inv : !felt.type, !felt.type
      %out = felt.add %mul, %const_1 : !felt.type, !felt.type
      struct.writem %self[@out] = %out : <@IsZero>, !felt.type
      function.return %self : !struct.type<@IsZero>
    }

    function.def @constrain(%self: !struct.type<@IsZero>, %in: !felt.type)
        attributes {function.allow_constraint} {
      %const_0 = felt.const 0
      %const_1 = felt.const 1
      %out = struct.readm %self[@out] : <@IsZero>, !felt.type
      %inv = struct.readm %self[@inv] : <@IsZero>, !felt.type
      %neg_in = felt.neg %in : !felt.type
      %mul = felt.mul %neg_in, %inv : !felt.type, !felt.type
      %sum = felt.add %mul, %const_1 : !felt.type, !felt.type
      constrain.eq %out, %sum : !felt.type, !felt.type
      %zero_prod = felt.mul %in, %out : !felt.type, !felt.type
      constrain.eq %zero_prod, %const_0 : !felt.type, !felt.type
      function.return
    }
  }
}
)mlir";

  static constexpr auto kProductFunctionIntervalModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @ProductIntervals {
    struct.member @out : !felt.type {llzk.pub, signal}

    function.def @product(%a: !felt.type) -> !struct.type<@ProductIntervals>
        attributes {function.allow_constraint, function.allow_non_native_field_ops, function.allow_witness, llzk.derived} {
      %self = struct.new : <@ProductIntervals>
      %five = felt.const 5
      %six = felt.const 6
      %cmp0 = bool.cmp ge(%a, %five) : !felt.type, !felt.type
      bool.assert %cmp0
      %cmp1 = bool.cmp le(%a, %six) : !felt.type, !felt.type
      bool.assert %cmp1
      %sum = felt.add %a, %five : !felt.type, !felt.type
      struct.writem %self[@out] = %sum : <@ProductIntervals>, !felt.type
      %read = struct.readm %self[@out] : <@ProductIntervals>, !felt.type
      constrain.eq %read, %sum : !felt.type, !felt.type
      function.return %self : !struct.type<@ProductIntervals>
    }

    function.def @compute(%a: !felt.type) -> !struct.type<@ProductIntervals>
        attributes {function.allow_witness} {
      %self = struct.new : <@ProductIntervals>
      function.return %self : !struct.type<@ProductIntervals>
    }

    function.def @constrain(%self: !struct.type<@ProductIntervals>, %a: !felt.type)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  static constexpr auto kProductFunctionUnreducedModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @ProductUnreducedIntervals {
    struct.member @out : !felt.type {llzk.pub, signal}

    function.def @product(%a: !felt.type) -> !struct.type<@ProductUnreducedIntervals>
        attributes {function.allow_constraint, function.allow_non_native_field_ops, function.allow_witness, llzk.derived} {
      %self = struct.new : <@ProductUnreducedIntervals>
      %five = felt.const 5
      %sum = felt.add %a, %five : !felt.type, !felt.type
      struct.writem %self[@out] = %sum : <@ProductUnreducedIntervals>, !felt.type
      %read = struct.readm %self[@out] : <@ProductUnreducedIntervals>, !felt.type
      constrain.eq %read, %sum : !felt.type, !felt.type
      function.return %self : !struct.type<@ProductUnreducedIntervals>
    }

    function.def @compute(%a: !felt.type) -> !struct.type<@ProductUnreducedIntervals>
        attributes {function.allow_witness} {
      %self = struct.new : <@ProductUnreducedIntervals>
      function.return %self : !struct.type<@ProductUnreducedIntervals>
    }

    function.def @constrain(%self: !struct.type<@ProductUnreducedIntervals>, %a: !felt.type)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  static constexpr auto kDynamicNonAssertCastModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @DynamicNonAssertCasts {
    struct.member @truncated : !felt.type<"babybear">
    struct.member @sum : !felt.type<"babybear">

    function.def @compute(%index: index, %felt: !felt.type<"babybear">)
        -> !struct.type<@DynamicNonAssertCasts>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@DynamicNonAssertCasts>
      %sat_to_felt = cast.tofelt sat %index : index, !felt.type<"babybear">
      %wrap_to_felt = cast.tofelt wrap %index : index, !felt.type<"babybear">
      %trunc_to_felt = cast.tofelt trunc %index : index, !felt.type<"babybear">
      %sat_to_index = cast.toindex sat %felt : !felt.type<"babybear">
      %wrap_to_index = cast.toindex wrap %felt : !felt.type<"babybear">
      %trunc_to_index = cast.toindex trunc %felt : !felt.type<"babybear">
      %one = felt.const 1 : !felt.type<"babybear">
      %sum = felt.add %trunc_to_felt, %one : !felt.type<"babybear">, !felt.type<"babybear">
      struct.writem %self[@truncated] = %trunc_to_felt : <@DynamicNonAssertCasts>, !felt.type<"babybear">
      struct.writem %self[@sum] = %sum : <@DynamicNonAssertCasts>, !felt.type<"babybear">
      function.return %self : !struct.type<@DynamicNonAssertCasts>
    }

    function.def @constrain(%self: !struct.type<@DynamicNonAssertCasts>, %index: index,
        %felt: !felt.type<"babybear">) attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  static constexpr auto kScfCarriedPodArrayModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @ScfCarriedPodArray {
    struct.member @out : !felt.type {llzk.pub, signal}

    function.def @compute() -> !struct.type<@ScfCarriedPodArray>
        attributes {function.allow_non_native_field_ops, function.allow_witness} {
      %self = struct.new : <@ScfCarriedPodArray>
      %three = felt.const 3
      %nine = felt.const 9
      %left = pod.new { @value = %three } : !pod.type<[@value: !felt.type]>
      %right = pod.new { @value = %nine } : !pod.type<[@value: !felt.type]>
      %pods = array.new %left, %right : <2 x !pod.type<[@value: !felt.type]>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %loop:2 = scf.while (%i = %c0, %state = %pods)
          : (index, !array.type<2 x !pod.type<[@value: !felt.type]>>)
          -> (index, !array.type<2 x !pod.type<[@value: !felt.type]>>) {
        %lt = arith.cmpi slt, %i, %c1 : index
        %eq = arith.cmpi eq, %i, %c0 : index
        %cond = bool.and %lt, %eq
        scf.condition(%cond) %i, %state
            : index, !array.type<2 x !pod.type<[@value: !felt.type]>>
      } do {
      ^bb0(%i: index, %state: !array.type<2 x !pod.type<[@value: !felt.type]>>):
        %pod = array.read %state[%c0]
            : <2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
        %value = pod.read %pod[@value] : !pod.type<[@value: !felt.type]>, !felt.type
        %next = arith.addi %i, %c1 : index
        scf.yield %next, %state
            : index, !array.type<2 x !pod.type<[@value: !felt.type]>>
      }
      %resultPod = array.read %loop#1[%c0]
          : <2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
      %result = pod.read %resultPod[@value] : !pod.type<[@value: !felt.type]>, !felt.type
      struct.writem %self[@out] = %result : <@ScfCarriedPodArray>, !felt.type
      function.return %self : !struct.type<@ScfCarriedPodArray>
    }

    function.def @constrain(%self: !struct.type<@ScfCarriedPodArray>)
        attributes {function.allow_constraint} {
      function.return
    }
  }
}
)mlir";

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
    EXPECT_TRUE(mod);
    return mod;
  }

  const IntervalAnalysisLattice *lookupLattice(ModuleIntervalAnalysis &analysis, Value value) {
    return analysis.getSolver().lookupState<IntervalAnalysisLattice>(value);
  }
};

// Scalar reads must use the same materialized aggregate roots as subsequent writes.
TEST_F(IntervalAnalysisAPITests, ScalarReadsUseMaterializedAggregateRoots) {
  for (bool useSwitch : {false, true}) {
    for (bool delayedWrites : {false, true}) {
      SCOPED_TRACE(useSwitch ? "index switch" : "if");
      SCOPED_TRACE(delayedWrites ? "reads before writes" : "writes before reads");
      std::string source = R"mlir(
!P = !pod.type<[@value: !felt.type]>
!A = !array.type<1 x !felt.type>
!W = !pod.type<[@pod: !P, @array: !A]>
module attributes {llzk.lang} {
  function.def @test(%condition: i1, %selector: index) -> (!felt.type, !felt.type, !felt.type, !felt.type) {
    %zero = arith.constant 0 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %pod = pod.new { @value = %seven } : !P
    %array = array.new %seven : !A
    // CONDITIONAL
    pod.write %selectedPod[@value] = %nine : !P, !felt.type
    array.write %selectedArray[%zero] = %nine : !A, !felt.type
    %copy = pod.new { @pod = %selectedPod, @array = %selectedArray } : !W
    %copiedPod = pod.read %copy[@pod] : !W, !P
    %copiedArray = pod.read %copy[@array] : !W, !A
    %podRead = pod.read %selectedPod[@value] : !P, !felt.type
    %arrayRead = array.read %selectedArray[%zero] : !A, !felt.type
    %nestedPodRead = pod.read %copiedPod[@value] : !P, !felt.type
    %nestedArrayRead = array.read %copiedArray[%zero] : !A, !felt.type
    function.return %podRead, %arrayRead, %nestedPodRead, %nestedArrayRead : !felt.type, !felt.type, !felt.type, !felt.type
  }
}
)mlir";
      source.replace(
          source.find("// CONDITIONAL"), std::string("// CONDITIONAL").size(),
          useSwitch ? R"mlir(
    %selectedPod, %selectedArray = scf.index_switch %selector -> !P, !A
    case 0 {
      scf.yield %pod, %array : !P, !A
    }
    default {
      scf.yield %pod, %array : !P, !A
    }
)mlir"
                    : R"mlir(
    %selectedPod, %selectedArray = scf.if %condition -> (!P, !A) {
      scf.yield %pod, %array : !P, !A
    } else {
      scf.yield %pod, %array : !P, !A
    }
)mlir"
      );
      auto mod = parseModule(source);
      ASSERT_TRUE(mod);
      const Field &field = Field::getField("babybear");
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);
      auto visit = [&](Operation *op) {
        SmallVector<const IntervalAnalysisLattice *> operands;
        SmallVector<IntervalAnalysisLattice *> results;
        for (Value value : op->getOperands()) {
          operands.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        for (Value value : op->getResults()) {
          results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        return analysis->visitOperation(op, operands, results);
      };
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      auto returned = fn.getBody().front().getTerminator()->getOperands();
      auto checkReads = [&](int64_t upper) {
        for (auto [index, value] : llvm::enumerate(returned)) {
          const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
          ASSERT_NE(lattice, nullptr);
          const auto &stored = lattice->getValue().getScalarValue();
          // Direct roots retain a flow-insensitive summary. Copies observe the last
          // guaranteed write, so their scalar reads have the exact replacement value.
          const int64_t lower = index < 2 ? 7 : upper;
          EXPECT_EQ(stored.getInterval(), UnreducedInterval(lower, upper).reduce(field));
          EXPECT_EQ(stored.getOptionalUnreducedInterval(), UnreducedInterval(lower, upper));
        }
      };
      SmallVector<Operation *> writes;
      auto *driver = solver.load<llzk::test::IntervalTestInitialization>([&] {
        for (Operation &op : fn.getBody().front()) {
          if (isa<pod::WritePodOp, array::WriteArrayOp>(op)) {
            writes.push_back(&op);
            if (delayedWrites) {
              continue;
            }
          }
          if (isa<scf::IfOp, scf::IndexSwitchOp>(op)) {
            for (Region &region : op.getRegions()) {
              ASSERT_TRUE(succeeded(visit(region.front().getTerminator())));
            }
          } else {
            ASSERT_TRUE(succeeded(visit(&op)));
          }
        }
        checkReads(delayedWrites ? 7 : 9);
        if (delayedWrites) {
          for (Operation *write : writes) {
            ASSERT_TRUE(succeeded(visit(write)));
          }
        }
      });
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod, [&](DataFlowAnalysis &candidate) {
        return &candidate == driver;
      })));
      checkReads(9);
    }
  }
}

// A published default must refine after an unconditional write, including through copies.
TEST_F(IntervalAnalysisAPITests, ConditionalPublicationsReplaceStaleDefaults) {
  for (bool delayedWrites : {false, true}) {
    for (bool independentWrite : {false, true}) {
      SCOPED_TRACE(delayedWrites ? "writes after publication" : "writes before publication");
      SCOPED_TRACE(independentWrite ? "independent result write" : "only conditional publication");
      std::string source = R"mlir(
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %seven = felt.const 7
    %nine = felt.const 9
    %index = arith.constant 0 : index
    %pod = pod.new : !pod.type<[@value: !felt.type]>
    %array = array.new : !array.type<1 x !felt.type>
    pod.write %pod[@value] = %seven : !pod.type<[@value: !felt.type]>, !felt.type
    array.write %array[%index] = %seven : !array.type<1 x !felt.type>, !felt.type
    %selectedPod, %selectedArray = scf.if %condition -> (!pod.type<[@value: !felt.type]>, !array.type<1 x !felt.type>) {
      scf.yield %pod, %array : !pod.type<[@value: !felt.type]>, !array.type<1 x !felt.type>
    } else {
      scf.yield %pod, %array : !pod.type<[@value: !felt.type]>, !array.type<1 x !felt.type>
    }
    // INDEPENDENT_WRITE
    %copy = pod.new { @pod = %selectedPod, @array = %selectedArray } : !pod.type<[@pod: !pod.type<[@value: !felt.type]>, @array: !array.type<1 x !felt.type>]>
    %out = array.new %copy : !array.type<1 x !pod.type<[@pod: !pod.type<[@value: !felt.type]>, @array: !array.type<1 x !felt.type>]>>
    function.return
  }
}
)mlir";
      if (independentWrite) {
        source.replace(
            source.find("// INDEPENDENT_WRITE"), std::string("// INDEPENDENT_WRITE").size(),
            "pod.write %selectedPod[@value] = %nine : !pod.type<[@value: !felt.type]>, !felt.type"
        );
      }
      auto mod = parseModule(source);
      ASSERT_TRUE(mod);
      const Field &field = Field::getField("babybear");
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);
      auto visit = [&](Operation *op) {
        SmallVector<const IntervalAnalysisLattice *> operands;
        SmallVector<IntervalAnalysisLattice *> results;
        for (Value value : op->getOperands()) {
          operands.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        for (Value value : op->getResults()) {
          results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        return analysis->visitOperation(op, operands, results);
      };
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      auto conditional = *fn.getOps<scf::IfOp>().begin();
      auto arrays = llvm::to_vector(fn.getOps<array::CreateArrayOp>());
      auto resultField = SourceRef(conditional->getResult(0))
                             .createChild(SourceRefIndex(StringAttr::get(&ctx, "value")));
      ASSERT_TRUE(succeeded(resultField));
      SmallVector<Operation *> writes;
      auto *driver = solver.load<llzk::test::IntervalTestInitialization>([&] {
        for (Operation &op : fn.getBody().front()) {
          if (&op == conditional.getOperation()) {
            break;
          }
          if (isa<pod::WritePodOp, array::WriteArrayOp>(op)) {
            writes.push_back(&op);
            if (delayedWrites) {
              continue;
            }
          }
          ASSERT_TRUE(succeeded(visit(&op)));
        }
        ASSERT_TRUE(succeeded(visit(conditional.thenBlock()->getTerminator())));
        ASSERT_TRUE(succeeded(visit(conditional.elseBlock()->getTerminator())));
        EXPECT_EQ(
            analysis->getWriteResults().at(*resultField).getInterval(),
            delayedWrites ? Interval::Entire(field) : Interval::Degenerate(field, field.felt(7))
        );
        // Publish the same initial state into both copies before refining the sources.
        for (Operation *op = conditional->getNextNode(); op != nullptr; op = op->getNextNode()) {
          ASSERT_TRUE(succeeded(visit(op)));
        }
        if (delayedWrites) {
          for (Operation *write : writes) {
            ASSERT_TRUE(succeeded(visit(write)));
          }
        }
      });
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod, [&](DataFlowAnalysis &candidate) {
        return &candidate == driver;
      })));
      auto check = [&](const SourceRef &ref, int64_t upper, bool copied = false) {
        const auto &stored = analysis->getWriteResults();
        auto it = stored.find(ref);
        ASSERT_NE(it, stored.end());
        // The result summary includes both publication and mutation; a later copy
        // contains only the value after the guaranteed independent write.
        const int64_t lower = copied ? upper : 7;
        EXPECT_EQ(it->second.getInterval(), UnreducedInterval(lower, upper).reduce(field));
        EXPECT_EQ(it->second.getOptionalUnreducedInterval(), UnreducedInterval(lower, upper));
      };
      check(*resultField, independentWrite ? 9 : 7);
      check(SourceRef(conditional->getResult(1), {SourceRefIndex(0)}), 7);
      check(
          SourceRef(
              arrays.back()->getResult(0),
              {SourceRefIndex(0), SourceRefIndex(StringAttr::get(&ctx, "pod")),
               SourceRefIndex(StringAttr::get(&ctx, "value"))}
          ),
          independentWrite ? 9 : 7, /*copied=*/true
      );
      check(
          SourceRef(
              arrays.back()->getResult(0),
              {SourceRefIndex(0), SourceRefIndex(StringAttr::get(&ctx, "array")), SourceRefIndex(0)}
          ),
          7
      );
    }
  }
}

namespace {

/// Count real interval transfer visits to test dependency invalidation independently of precision.
class CountingIntervalAnalysis : public IntervalDataFlowAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountingIntervalAnalysis)
  using IntervalDataFlowAnalysis::IntervalDataFlowAnalysis;

  /// Count each scheduled transfer before running the normal analysis implementation.
  LogicalResult visitOperation(
      Operation *op, ArrayRef<const IntervalAnalysisLattice *> operands,
      ArrayRef<IntervalAnalysisLattice *> results
  ) override {
    ++visits[op];
    return IntervalDataFlowAnalysis::visitOperation(op, operands, results);
  }

  /// Return the number of transfers performed for an operation.
  unsigned getVisits(Operation *op) const { return visits.lookup(op); }

private:
  llvm::DenseMap<Operation *, unsigned> visits;
};

} // namespace

// Ranged subscriptions must observe writes through concrete aliases of the same storage.
TEST_F(IntervalAnalysisAPITests, OverlappingArrayWritesNotifyRangedSubscribers) {
  for (unsigned sourceKind : {0U, 1U, 2U}) {
    for (bool rangedWrite : {false, true}) {
      SCOPED_TRACE(sourceKind); // Argument, owned array, and array nested in a POD.
      SCOPED_TRACE(rangedWrite ? "ranged write" : "concrete write");
      auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%index: index, %condition: i1, %argument: !array.type<2 x !pod.type<[@value: !felt.type]>>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %seed = pod.new { @value = %seven } : !pod.type<[@value: !felt.type]>
    %owned = array.new : !array.type<2 x !pod.type<[@value: !felt.type]>>
    %container = pod.new { @items = %owned } : !pod.type<[@items: !array.type<2 x !pod.type<[@value: !felt.type]>>]>
    %nested = pod.read %container[@items] : !pod.type<[@items: !array.type<2 x !pod.type<[@value: !felt.type]>>]>, !array.type<2 x !pod.type<[@value: !felt.type]>>
    %other = array.new : !array.type<2 x !pod.type<[@value: !felt.type]>>
    array.write %argument[%zero] = %seed : !array.type<2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
    array.write %argument[%one] = %seed : !array.type<2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
    %fixed = array.read %argument[%one] : !array.type<2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
    pod.write %fixed[@value] = %nine : !pod.type<[@value: !felt.type]>, !felt.type
    array.write %argument[%one] = %fixed : !array.type<2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
    %slice = array.read %argument[%index] : !array.type<2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
    %copy = pod.new { @slice = %slice } : !pod.type<[@slice: !pod.type<[@value: !felt.type]>]>
    %otherSlice = array.read %other[%index] : !array.type<2 x !pod.type<[@value: !felt.type]>>, !pod.type<[@value: !felt.type]>
    %unrelated = pod.new { @slice = %otherSlice } : !pod.type<[@slice: !pod.type<[@value: !felt.type]>]>
    %selected = scf.if %condition -> (!pod.type<[@value: !felt.type]>) {
      scf.yield %slice : !pod.type<[@value: !felt.type]>
    } else {
      scf.yield %slice : !pod.type<[@value: !felt.type]>
    }
    function.return
  }
}
)mlir");
      ASSERT_TRUE(mod);
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      auto arrays = llvm::to_vector(fn.getOps<array::CreateArrayOp>());
      auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
      auto writes = llvm::to_vector(fn.getOps<array::WriteArrayOp>());
      auto write = writes.back();
      auto selections = llvm::to_vector(fn.getOps<array::ReadArrayOp>());
      auto fixed = selections.front();
      auto read = selections[1];
      auto conditional = *fn.getOps<scf::IfOp>().begin();
      auto nested = *fn.getOps<pod::ReadPodOp>().begin();
      std::array<Value, 3> sources {
          fn.getArgument(2), arrays.front().getResult(), nested.getResult()
      };
      read->setOperand(0, sources[sourceKind]);
      fixed->setOperand(0, sources[sourceKind]);
      for (auto stored : writes) {
        stored->setOperand(0, sources[sourceKind]);
      }
      if (rangedWrite) {
        fixed->setOperand(1, fn.getArgument(0));
        write->setOperand(1, fn.getArgument(0));
      }

      const Field &field = Field::getField("babybear");
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      auto *analysis = solver.load<CountingIntervalAnalysis>(smt, field, false, false);
      auto visit = [&](Operation *op) {
        SmallVector<const IntervalAnalysisLattice *> operands;
        SmallVector<IntervalAnalysisLattice *> results;
        for (Value value : op->getOperands()) {
          operands.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        for (Value value : op->getResults()) {
          results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        return analysis->visitOperation(op, operands, results);
      };
      auto *thenYield = conditional.thenBlock()->getTerminator();
      auto *elseYield = conditional.elseBlock()->getTerminator();
      auto *driver = solver.load<llzk::test::IntervalTestInitialization>([&] {
        // Initialize roots before subscribing ranged consumers: pending-producer root
        // subscriptions would otherwise conceal the missing overlapping-path dependency.
        for (Operation &op : fn.getBody().front()) {
          if (&op == write.getOperation()) {
            continue;
          }
          if (&op == conditional.getOperation()) {
            ASSERT_TRUE(succeeded(visit(thenYield)));
            ASSERT_TRUE(succeeded(visit(elseYield)));
          } else {
            ASSERT_TRUE(succeeded(visit(&op)));
          }
        }
        EXPECT_EQ(analysis->getVisits(pods[2]), 1U);
        EXPECT_EQ(analysis->getVisits(thenYield), 1U);
        EXPECT_EQ(analysis->getVisits(elseYield), 1U);
        ASSERT_TRUE(succeeded(visit(write)));
      });
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod, [&](DataFlowAnalysis &candidate) {
        return &candidate == driver;
      })));
      EXPECT_GT(analysis->getVisits(pods[2]), 1U);
      EXPECT_GT(analysis->getVisits(thenYield), 1U);
      EXPECT_GT(analysis->getVisits(elseYield), 1U);
      EXPECT_EQ(analysis->getVisits(pods[3]), 1U);
    }
  }
}

// A ranged write must widen concrete copies and yields regardless of visitation order.
TEST_F(IntervalAnalysisAPITests, RangedWritesUpdateConcreteAggregateSubscribers) {
  for (bool delayedWrite : {false, true}) {
    for (bool aggregateWrite : {false, true}) {
      SCOPED_TRACE(delayedWrite ? "visit write after copies" : "visit write before copies");
      SCOPED_TRACE(aggregateWrite ? "aggregate write" : "scalar write");
      std::string source = R"mlir(
module attributes {llzk.lang} {
  function.def @test(%index: index, %condition: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %payload = pod.new { @value = %seven } : !pod.type<[@value: !felt.type]>
    %replacement = pod.new { @value = %nine } : !pod.type<[@value: !felt.type]>
    %element = pod.new { @payload = %payload, @other = %seven } : !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    %array = array.new %element, %element, %element, %element : !array.type<2,2 x !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>
    %dynamic = array.read %array[%zero, %index] : !array.type<2,2 x !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>, !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    %dynamicPayload = pod.read %dynamic[@payload] : !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>, !pod.type<[@value: !felt.type]>
    // LATE_WRITE
    %concrete = array.read %array[%zero, %one] : !array.type<2,2 x !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>, !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    %podCopy = pod.new { @selected = %concrete } : !pod.type<[@selected: !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>]>
    %arrayCopy = array.new %concrete : !array.type<1 x !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>
    %assigned = pod.new : !pod.type<[@selected: !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>]>
    pod.write %assigned[@selected] = %concrete : !pod.type<[@selected: !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>]>, !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    %selected = scf.if %condition -> (!pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>) {
      scf.yield %concrete : !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    } else {
      scf.yield %concrete : !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    }
    %out = array.new %selected : !array.type<1 x !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>
    %disjoint = array.read %array[%one, %one] : !array.type<2,2 x !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>, !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>
    %unrelated = pod.new { @selected = %disjoint } : !pod.type<[@selected: !pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>]>
    function.return
  }
}
)mlir";
      source.replace(
          source.find("// LATE_WRITE"), std::string("// LATE_WRITE").size(),
          aggregateWrite
              ? "pod.write %dynamic[@payload] = %replacement : !pod.type<[@payload: "
                "!pod.type<[@value: !felt.type]>, @other: !felt.type]>, !pod.type<[@value: "
                "!felt.type]>\n"
                "array.write %array[%zero, %index] = %dynamic : !array.type<2,2 x "
                "!pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>, "
                "!pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>"
              : "pod.write %dynamicPayload[@value] = %nine : !pod.type<[@value: !felt.type]>, "
                "!felt.type\n"
                "pod.write %dynamic[@payload] = %dynamicPayload : !pod.type<[@payload: "
                "!pod.type<[@value: !felt.type]>, @other: !felt.type]>, !pod.type<[@value: "
                "!felt.type]>\n"
                "array.write %array[%zero, %index] = %dynamic : !array.type<2,2 x "
                "!pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>>, "
                "!pod.type<[@payload: !pod.type<[@value: !felt.type]>, @other: !felt.type]>"
      );
      auto mod = parseModule(source);
      ASSERT_TRUE(mod);
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
      auto arrays = llvm::to_vector(fn.getOps<array::CreateArrayOp>());
      auto conditional = *fn.getOps<scf::IfOp>().begin();
      auto concrete = llvm::to_vector(fn.getOps<array::ReadArrayOp>())[1];
      auto payloadRead = *fn.getOps<pod::ReadPodOp>().begin();
      SmallVector<Operation *> writes;
      for (Operation *writer = payloadRead->getNextNode(); writer != concrete.getOperation();
           writer = writer->getNextNode()) {
        writes.push_back(writer);
      }
      const Field &field = Field::getField("babybear");
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      auto *analysis = solver.load<CountingIntervalAnalysis>(smt, field, false, true);
      auto visit = [&](Operation *op) {
        SmallVector<const IntervalAnalysisLattice *> operands;
        SmallVector<IntervalAnalysisLattice *> results;
        for (Value value : op->getOperands()) {
          operands.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        for (Value value : op->getResults()) {
          results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
        }
        return analysis->visitOperation(op, operands, results);
      };
      SourceRefIndex selected(StringAttr::get(&ctx, "selected"));
      SourceRefIndex payload(StringAttr::get(&ctx, "payload"));
      SourceRefIndex value(StringAttr::get(&ctx, "value"));
      SourceRefIndex other(StringAttr::get(&ctx, "other"));
      SmallVector<SourceRef> copies {
          SourceRef(pods[3]->getResult(0), {selected}),
          SourceRef(arrays[1]->getResult(0), {SourceRefIndex(0)}),
          SourceRef(pods[4]->getResult(0), {selected}), SourceRef(conditional->getResult(0)),
          SourceRef(arrays[2]->getResult(0), {SourceRefIndex(0)})
      };
      auto check = [&](const SourceRef &ref, int64_t upper) {
        auto it = analysis->getWriteResults().find(ref);
        ASSERT_NE(it, analysis->getWriteResults().end());
        EXPECT_EQ(it->second.getInterval(), UnreducedInterval(7, upper).reduce(field));
        EXPECT_EQ(it->second.getOptionalUnreducedInterval(), UnreducedInterval(7, upper));
      };
      auto checkCopies = [&](int64_t upper) {
        for (const SourceRef &copy : copies) {
          check(*copy.createChild(payload)->createChild(value), upper);
          check(*copy.createChild(other), 7);
        }
      };
      auto visitWrites = [&] {
        for (Operation *writer : writes) {
          ASSERT_TRUE(succeeded(visit(writer)));
        }
      };
      auto *driver = solver.load<llzk::test::IntervalTestInitialization>([&] {
        for (Operation &op : fn.getBody().front()) {
          if (&op == concrete.getOperation() && !delayedWrite) {
            visitWrites();
          }
          if (llvm::is_contained(writes, &op)) {
            continue;
          }
          if (&op == conditional.getOperation()) {
            ASSERT_TRUE(succeeded(visit(conditional.thenBlock()->getTerminator())));
            ASSERT_TRUE(succeeded(visit(conditional.elseBlock()->getTerminator())));
          } else {
            ASSERT_TRUE(succeeded(visit(&op)));
          }
        }
        checkCopies(delayedWrite ? 7 : 9);
        if (delayedWrite) {
          visitWrites();
        }
      });
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod, [&](DataFlowAnalysis &candidate) {
        return &candidate == driver;
      })));
      checkCopies(9);
      check(SourceRef(pods[5]->getResult(0), {selected, payload, value}), 7);
      EXPECT_EQ(analysis->getVisits(pods[5]), 1U);
    }
  }
}

// A join can run before aggregate initializers when the worklist discovers new live regions.
TEST_F(IntervalAnalysisAPITests, DelayedAggregateFieldsRevisitConditionalYield) {
  auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %zero = felt.const 0
    %seven = felt.const 7
    %left = pod.new { @value = %zero } : !pod.type<[@value: !felt.type]>
    %right = pod.new { @value = %seven } : !pod.type<[@value: !felt.type]>
    %selected = scf.if %condition -> (!pod.type<[@value: !felt.type]>) {
      scf.yield %left : !pod.type<[@value: !felt.type]>
    } else {
      scf.yield %right : !pod.type<[@value: !felt.type]>
    }
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  const Field &field = Field::getField("babybear");
  auto smt = llvm::CreateZ3Solver();
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  llzk::dataflow::loadRequiredAnalyses(solver);
  solver.load<SourceRefAnalysis>();
  ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
  auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);

  // Settle liveness and storage identities first, then explicitly exercise the legal
  // worklist order where both yields precede the writes that materialize their sources.
  auto visit = [&](Operation *op) {
    SmallVector<const IntervalAnalysisLattice *> operands;
    SmallVector<IntervalAnalysisLattice *> results;
    for (Value value : op->getOperands()) {
      operands.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
    }
    for (Value value : op->getResults()) {
      results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
    }
    return analysis->visitOperation(op, operands, results);
  };
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  auto conditional = *fn.getOps<scf::IfOp>().begin();
  auto resultField = SourceRef(conditional->getResult(0))
                         .createChild(SourceRefIndex(StringAttr::get(&ctx, "value")));
  ASSERT_TRUE(succeeded(resultField));
  auto *driver = solver.load<llzk::test::IntervalTestInitialization>([&] {
    ASSERT_TRUE(succeeded(visit(conditional.thenBlock()->getTerminator())));
    ASSERT_TRUE(succeeded(visit(conditional.elseBlock()->getTerminator())));
    EXPECT_FALSE(analysis->getWriteResults().contains(*resultField));

    for (Operation &op : fn.getBody().front()) {
      if (isa<felt::FeltConstantOp, pod::NewPodOp>(op)) {
        ASSERT_TRUE(succeeded(visit(&op)));
      }
    }
  });
  // Drain only queued dependencies. Reinitializing the analysis would revisit every yield
  // and mask a missing storage subscription.
  ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod, [&](DataFlowAnalysis &candidate) {
    return &candidate == driver;
  })));
  const auto &writes = analysis->getWriteResults();
  auto stored = writes.find(*resultField);
  ASSERT_NE(stored, writes.end());
  EXPECT_EQ(stored->second.getInterval(), UnreducedInterval(0, 7).reduce(field));
  EXPECT_EQ(stored->second.getOptionalUnreducedInterval(), UnreducedInterval(0, 7));
}

// Copies made before a conditional publishes its fields must update through every consumer.
TEST_F(IntervalAnalysisAPITests, DelayedAggregateFieldsPropagateThroughCopyChains) {
  auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %zero = felt.const 0
    %seven = felt.const 7
    %left = pod.new { @value = %zero } : !pod.type<[@value: !felt.type]>
    %right = pod.new { @value = %seven } : !pod.type<[@value: !felt.type]>
    %selected = scf.if %condition -> (!pod.type<[@value: !felt.type]>) {
      scf.yield %left : !pod.type<[@value: !felt.type]>
    } else {
      scf.yield %right : !pod.type<[@value: !felt.type]>
    }
    %copy1 = pod.new { @child = %selected } : !pod.type<[@child: !pod.type<[@value: !felt.type]>]>
    %alias = pod.read %copy1[@child] : !pod.type<[@child: !pod.type<[@value: !felt.type]>]>, !pod.type<[@value: !felt.type]>
    %copy2 = pod.new { @child = %alias } : !pod.type<[@child: !pod.type<[@value: !felt.type]>]>
    %array = array.new %copy2, %copy2 : !array.type<2 x !pod.type<[@child: !pod.type<[@value: !felt.type]>]>>
    %copy3 = pod.new { @values = %array } : !pod.type<[@values: !array.type<2 x !pod.type<[@child: !pod.type<[@value: !felt.type]>]>>]>
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  const Field &field = Field::getField("babybear");
  auto smt = llvm::CreateZ3Solver();
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  llzk::dataflow::loadRequiredAnalyses(solver);
  solver.load<SourceRefAnalysis>();
  ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
  auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);
  auto visit = [&](Operation *op) {
    SmallVector<const IntervalAnalysisLattice *> operands;
    SmallVector<IntervalAnalysisLattice *> results;
    for (Value value : op->getOperands()) {
      operands.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
    }
    for (Value value : op->getResults()) {
      results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
    }
    return analysis->visitOperation(op, operands, results);
  };

  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  auto conditional = *fn.getOps<scf::IfOp>().begin();
  auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
  auto *driver = solver.load<llzk::test::IntervalTestInitialization>([&] {
    // Visit the entire consumer chain first, so every later copy needs a storage notification.
    for (Operation *op = conditional->getNextNode(); op != nullptr; op = op->getNextNode()) {
      ASSERT_TRUE(succeeded(visit(op)));
    }
    ASSERT_TRUE(succeeded(visit(conditional.thenBlock()->getTerminator())));
    ASSERT_TRUE(succeeded(visit(conditional.elseBlock()->getTerminator())));
    for (Operation &op : fn.getBody().front()) {
      if (&op == conditional.getOperation()) {
        break;
      }
      ASSERT_TRUE(succeeded(visit(&op)));
    }
  });
  ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod, [&](DataFlowAnalysis &candidate) {
    return &candidate == driver;
  })));

  const auto &writes = analysis->getWriteResults();
  for (int64_t index : {0, 1}) {
    SourceRef leaf(
        pods.back()->getResult(0),
        {SourceRefIndex(StringAttr::get(&ctx, "values")), SourceRefIndex(index),
         SourceRefIndex(StringAttr::get(&ctx, "child")),
         SourceRefIndex(StringAttr::get(&ctx, "value"))}
    );
    auto stored = writes.find(leaf);
    ASSERT_NE(stored, writes.end());
    EXPECT_EQ(stored->second.getInterval(), UnreducedInterval(0, 7).reduce(field));
    EXPECT_EQ(stored->second.getOptionalUnreducedInterval(), UnreducedInterval(0, 7));
  }
}

// Equal ranges describe possible locations, not equal values at independent dynamic indices.
TEST_F(IntervalAnalysisAPITests, AmbiguousArrayReadsHaveIndependentSymbols) {
  for (bool storedRange : {false, true}) {
    SCOPED_TRACE(storedRange ? "ranged writes" : "unwritten arrays");
    std::string source = R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !felt.type>, %pods: !array.type<2 x !P>, %i: index, %j: index, %input: !felt.type, %condition: i1) -> (!felt.type, !felt.type, !felt.type, !felt.type, !felt.type, !felt.type) {
    %leftPod = array.read %pods[%i] : !array.type<2 x !P>, !P
    %rightPod = array.read %pods[%j] : !array.type<2 x !P>, !P
    // WRITES
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %array[%j] : !array.type<2 x !felt.type>, !felt.type
    %leftCopy = pod.new { @child = %leftPod } : !W
    %rightCopy = pod.new { @child = %rightPod } : !W
    %leftChild = pod.read %leftCopy[@child] : !W, !P
    %rightChild = pod.read %rightCopy[@child] : !W, !P
    %copiedLeft = pod.read %leftChild[@value] : !P, !felt.type
    %copiedRight = pod.read %rightChild[@value] : !P, !felt.type
    %leftChoice = scf.if %condition -> (!P) {
      scf.yield %leftPod : !P
    } else {
      scf.yield %leftPod : !P
    }
    %rightChoice = scf.if %condition -> (!P) {
      scf.yield %rightPod : !P
    } else {
      scf.yield %rightPod : !P
    }
    %chosenLeft = pod.read %leftChoice[@value] : !P, !felt.type
    %chosenRight = pod.read %rightChoice[@value] : !P, !felt.type
    function.return %left, %right, %copiedLeft, %copiedRight, %chosenLeft, %chosenRight : !felt.type, !felt.type, !felt.type, !felt.type, !felt.type, !felt.type
  }
}
)mlir";
    if (storedRange) {
      source.replace(
          source.find("// WRITES"), std::string("// WRITES").size(),
          "array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type\n"
          "    pod.write %leftPod[@value] = %input : !P, !felt.type"
      );
    }
    auto mod = parseModule(source);
    ASSERT_TRUE(mod);
    const Field &field = Field::getField("babybear");
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);
    ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto returned = fn.getBody().front().getTerminator()->getOperands();
    SmallVector<llvm::SMTExprRef> expressions;
    for (Value value : returned) {
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
      ASSERT_NE(lattice, nullptr);
      const auto &stored = lattice->getValue().getScalarValue();
      ASSERT_NE(stored.getExpr(), nullptr);
      EXPECT_EQ(stored.getInterval(), Interval::Entire(field));
      EXPECT_EQ(
          stored.getOptionalUnreducedInterval(), UnreducedInterval(field.zero(), field.maxVal())
      );
      expressions.push_back(stored.getExpr());
    }
    for (unsigned first : {0U, 2U, 4U}) {
      EXPECT_FALSE(*expressions[first] == *expressions[first + 1]);
      if (first == 0) {
        EXPECT_EQ(expressions[first], analysis->getOrCreateSymbol(returned[first]));
        EXPECT_EQ(expressions[first + 1], analysis->getOrCreateSymbol(returned[first + 1]));
      }
      // Distinct dynamic selections can read 7 and 9, even after a ranged write.
      smt->addConstraint(smt->mkEqual(
          expressions[first], smt->mkBitvector(toAPSInt(field.felt(7)), field.bitWidth())
      ));
      smt->addConstraint(smt->mkEqual(
          expressions[first + 1], smt->mkBitvector(toAPSInt(field.felt(9)), field.bitWidth())
      ));
    }
    EXPECT_EQ(smt->check(), std::optional<bool>(true));
  }
}

// Conditional storage roots retain the nested/disjoint alias proofs from the preceding stack.
TEST_F(IntervalAnalysisAPITests, ConditionalArrayReadsPreserveStackedAliasProofs) {
  for (bool sameIndex : {false, true}) {
    for (bool disjointWrite : {false, true}) {
      SCOPED_TRACE(sameIndex ? "same index" : "independent indices");
      SCOPED_TRACE(disjointWrite ? "disjoint write" : "overlapping write");
      std::string source = R"mlir(
!A = !array.type<2 x !felt.type>
module attributes {llzk.lang} {
  function.def @test(%a: !A, %b: !A, %i: index, %j: index, %condition: i1, %input: !felt.type) {
    %selected = scf.if %condition -> (!A) {
      scf.yield %a : !A
    } else {
      scf.yield %b : !A
    }
    %fresh = array.new : !A
    %yes = arith.constant true
    %left = array.read %selected[%i] : !A, !felt.type
    array.write WRITE_TARGET[%i] = %input : !A, !felt.type
    scf.if %yes {
      %right = array.read %selected[READ_INDEX] : !A, !felt.type
      %different = bool.cmp ne(%left, %right)
      bool.assert %different
    }
    function.return
  }
}
)mlir";
      source.replace(
          source.find("WRITE_TARGET"), StringRef("WRITE_TARGET").size(),
          disjointWrite ? "%fresh" : "%selected"
      );
      source.replace(
          source.find("READ_INDEX"), StringRef("READ_INDEX").size(), sameIndex ? "%i" : "%j"
      );
      auto mod = parseModule(source);
      ASSERT_TRUE(mod);
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      solver.load<IntervalDataFlowAnalysis>(smt, Field::getField("babybear"), false, false);
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      SmallVector<boolean::CmpOp> comparisons;
      mod->walk([&](boolean::CmpOp comparison) { comparisons.push_back(comparison); });
      ASSERT_EQ(comparisons.size(), 1U);
      const auto *lattice =
          solver.lookupState<IntervalAnalysisLattice>(comparisons.front().getResult());
      ASSERT_NE(lattice, nullptr);
      ASSERT_FALSE(lattice->getConstraints().empty());
      for (const ExpressionValue &constraint : lattice->getConstraints()) {
        smt->addConstraint(constraint.getExpr());
      }
      EXPECT_EQ(smt->check(), std::optional<bool>(!(sameIndex && disjointWrite)));
    }
  }
}

// Equal-looking values in separate functions must still have separate SMT identities.
TEST_F(IntervalAnalysisAPITests, ValueSymbolsCacheIdentityAndPreserveSorts) {
  auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @first(%felt: !felt.type, %flag: i1, %index: index,
                      %array: !array.type<2 x !felt.type>) {
    function.return
  }
  function.def @second(%felt: !felt.type, %flag: i1, %index: index,
                       %array: !array.type<2 x !felt.type>) {
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto functions = llvm::to_vector(mod->getOps<function::FuncDefOp>());
  ASSERT_EQ(functions.size(), 2U);
  const Field &field = Field::getField("babybear");
  auto smt = llvm::CreateZ3Solver();
  DataFlowSolver solver;
  auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);

  for (auto [first, second] : llvm::zip(functions[0].getArguments(), functions[1].getArguments())) {
    EXPECT_EQ(buildStringViaPrint(first), buildStringViaPrint(second));
    const auto *symbol = analysis->getOrCreateSymbol(first);
    EXPECT_EQ(symbol, analysis->getOrCreateSymbol(first));
    const auto *other = analysis->getOrCreateSymbol(second);
    EXPECT_FALSE(*symbol == *other);
    EXPECT_EQ(symbol, analysis->getOrCreateSymbol(first));
    const auto *sort = smt->getSort(symbol);
    if (first.getType().isInteger(1)) {
      EXPECT_TRUE(sort->isBooleanSort());
    } else {
      ASSERT_TRUE(sort->isBitvectorSort());
      EXPECT_EQ(sort->getBitvectorSortSize(), field.bitWidth());
    }

    // Even a user-supplied SourceRef name matching the internal value name is separate.
    functions[0].setArgName(first.getArgNumber(), buildStringViaPrint(*symbol));
    SourceRef ref(first);
    EXPECT_EQ(buildStringViaPrint(ref), buildStringViaPrint(*symbol));
    const auto *refSymbol = analysis->getOrCreateSymbol(ref);
    EXPECT_EQ(refSymbol, analysis->getOrCreateSymbol(ref));
    EXPECT_FALSE(*symbol == *refSymbol);
    EXPECT_TRUE(*sort == *smt->getSort(refSymbol));
  }
}

// Printed references may coincide across functions, including paths into aggregate arguments.
TEST_F(IntervalAnalysisAPITests, SourceRefSymbolsCacheIdentityAndPreserveSorts) {
  auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @first(%felt: !felt.type, %flag: i1, %index: index,
                      %array: !array.type<2 x !felt.type>) {
    function.return
  }
  function.def @second(%felt: !felt.type, %flag: i1, %index: index,
                       %array: !array.type<2 x !felt.type>) {
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto functions = llvm::to_vector(mod->getOps<function::FuncDefOp>());
  ASSERT_EQ(functions.size(), 2U);
  const Field &field = Field::getField("babybear");
  auto smt = llvm::CreateZ3Solver();
  DataFlowSolver solver;
  auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);

  llvm::SmallVector<SourceRef> firstRefs, secondRefs;
  for (auto [first, second] : llvm::zip(functions[0].getArguments(), functions[1].getArguments())) {
    firstRefs.emplace_back(first);
    secondRefs.emplace_back(second);
  }
  for (int64_t index : {0, 1}) {
    auto first = SourceRef(functions[0].getArgument(3)).createChild(SourceRefIndex(index));
    auto second = SourceRef(functions[1].getArgument(3)).createChild(SourceRefIndex(index));
    ASSERT_TRUE(succeeded(first));
    ASSERT_TRUE(succeeded(second));
    firstRefs.push_back(*first);
    secondRefs.push_back(*second);
  }
  for (auto [first, second] : llvm::zip(firstRefs, secondRefs)) {
    EXPECT_NE(first, second);
    EXPECT_EQ(buildStringViaPrint(first), buildStringViaPrint(second));
    const auto *symbol = analysis->getOrCreateSymbol(first);
    const auto *other = analysis->getOrCreateSymbol(second);
    EXPECT_EQ(symbol, analysis->getOrCreateSymbol(SourceRef(first)));
    EXPECT_FALSE(*symbol == *other);
    const auto *sort = smt->getSort(symbol);
    EXPECT_TRUE(*sort == *smt->getSort(other));
    if (first.getType().isInteger(1)) {
      EXPECT_TRUE(sort->isBooleanSort());
    } else {
      ASSERT_TRUE(sort->isBitvectorSort());
      EXPECT_EQ(sort->getBitvectorSortSize(), field.bitWidth());
    }
    smt->addConstraint(smt->mkNot(smt->mkEqual(symbol, other)));
  }
  EXPECT_EQ(smt->check(), std::optional<bool>(true));
}

// Namespace allocation outlives an analysis when another owner retains its SMT solver.
TEST_F(IntervalAnalysisAPITests, ValueSymbolsRemainDistinctAcrossAnalysesSharingSolver) {
  auto mod = parseModule(kDynamicNonAssertCastModule);
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto value = structDef.getComputeFuncOp().getArguments().front();
  const Field &field = Field::getField("babybear");
  auto smt = llvm::CreateZ3Solver();
  llvm::SMTExprRef firstSymbol, firstRef;
  {
    DataFlowSolver firstSolver, secondSolver;
    auto *first = firstSolver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    auto *second = secondSolver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    firstSymbol = first->getOrCreateSymbol(value);
    firstRef = first->getOrCreateSymbol(SourceRef(value));
    const auto *secondSymbol = second->getOrCreateSymbol(value);
    const auto *secondRef = second->getOrCreateSymbol(SourceRef(value));
    EXPECT_FALSE(*firstSymbol == *secondSymbol);
    EXPECT_FALSE(*firstSymbol == *secondRef);
    EXPECT_FALSE(*firstRef == *secondSymbol);
    EXPECT_FALSE(*firstRef == *secondRef);
    EXPECT_EQ(firstSymbol, first->getOrCreateSymbol(value));
    EXPECT_EQ(secondSymbol, second->getOrCreateSymbol(value));
    smt->addConstraint(smt->mkNot(smt->mkEqual(firstSymbol, secondSymbol)));
    EXPECT_EQ(smt->check(), std::optional<bool>(true));
  }

  // Solver reset removes constraints, but existing symbol names remain allocated.
  smt->reset();
  {
    DataFlowSolver solver;
    auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    const auto *symbol = analysis->getOrCreateSymbol(value);
    EXPECT_FALSE(*firstSymbol == *symbol);
    EXPECT_FALSE(*firstRef == *symbol);
    EXPECT_FALSE(*firstRef == *analysis->getOrCreateSymbol(SourceRef(value)));
  }
  std::weak_ptr<llvm::SMTSolver> weakSolver = smt;
  smt.reset();
  EXPECT_TRUE(weakSolver.expired());
}

// Inverse equations must remain independent for identical operations in different contexts.
TEST_F(IntervalAnalysisAPITests, InverseSymbolsRemainDistinctAcrossFunctionsAndAnalyses) {
  auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @first(%value: !felt.type) -> (!felt.type, !felt.type)
      attributes {function.allow_witness, function.allow_non_native_field_ops} {
    %one = felt.const 1 : !felt.type
    %inverse = felt.inv %value : !felt.type
    %quotient = felt.div %one, %value : !felt.type, !felt.type
    function.return %inverse, %quotient : !felt.type, !felt.type
  }
  function.def @second(%value: !felt.type) -> (!felt.type, !felt.type)
      attributes {function.allow_witness, function.allow_non_native_field_ops} {
    %one = felt.const 1 : !felt.type
    %inverse = felt.inv %value : !felt.type
    %quotient = felt.div %one, %value : !felt.type, !felt.type
    function.return %inverse, %quotient : !felt.type, !felt.type
  }
}
)mlir");
  ASSERT_TRUE(mod);
  llvm::SmallVector<felt::InvFeltOp> invOps;
  llvm::SmallVector<felt::DivFeltOp> divOps;
  mod->walk([&](felt::InvFeltOp op) { invOps.push_back(op); });
  mod->walk([&](felt::DivFeltOp op) { divOps.push_back(op); });
  ASSERT_EQ(invOps.size(), 2U);
  ASSERT_EQ(divOps.size(), 2U);
  EXPECT_EQ(buildStringViaInsertionOp(*invOps[0]), buildStringViaInsertionOp(*invOps[1]));
  EXPECT_EQ(buildStringViaInsertionOp(*divOps[0]), buildStringViaInsertionOp(*divOps[1]));

  const Field &field = Field::getField("babybear");
  for (bool separateAnalyses : {false, true}) {
    SCOPED_TRACE(
        separateAnalyses ? "shared solver, same operation" : "same analysis, two functions"
    );
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver first(DataFlowConfig().setInterprocedural(false));
    DataFlowSolver second(DataFlowConfig().setInterprocedural(false));
    llvm::SmallVector<llvm::SMTExprRef> inputs, inverses, quotients;
    for (unsigned index = 0; index < 2; ++index) {
      auto &solver = separateAnalyses && index == 1 ? second : first;
      if (index == 0 || separateAnalyses) {
        llzk::dataflow::loadRequiredAnalyses(solver);
        solver.load<SourceRefAnalysis>();
        solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
        ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      }
      unsigned opIndex = separateAnalyses ? 0 : index;
      auto getExpr = [&](Value value) -> llvm::SMTExprRef {
        const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
        EXPECT_NE(lattice, nullptr);
        return lattice ? lattice->getValue().getScalarValue().getExpr() : nullptr;
      };
      inputs.push_back(getExpr(invOps[opIndex]->getOperand(0)));
      inverses.push_back(getExpr(invOps[opIndex].getResult()));
      quotients.push_back(getExpr(divOps[opIndex].getResult()));
      ASSERT_NE(inputs.back(), nullptr);
      ASSERT_NE(inverses.back(), nullptr);
      ASSERT_NE(quotients.back(), nullptr);

      // Bind the symbolic inputs only after analysis, so constant folding cannot hide a collision.
      auto input = field.felt(static_cast<int>(index) + 1);
      const auto *expectedInverse = smt->mkBitvector(toAPSInt(field.inv(input)), field.bitWidth());
      smt->addConstraint(
          smt->mkEqual(inputs.back(), smt->mkBitvector(toAPSInt(input), field.bitWidth()))
      );
      smt->addConstraint(smt->mkEqual(inverses.back(), expectedInverse));
      smt->addConstraint(smt->mkEqual(quotients.back(), expectedInverse));
    }
    EXPECT_FALSE(*inputs[0] == *inputs[1]);
    EXPECT_FALSE(*inverses[0] == *inverses[1]);
    EXPECT_FALSE(*quotients[0] == *quotients[1]);
    EXPECT_EQ(smt->check(), std::optional<bool>(true));
  }
}

TEST_F(IntervalAnalysisAPITests, DynamicNonAssertCastsInitializeResultLattices) {
  auto mod = parseModule(kDynamicNonAssertCastModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  function::FuncDefOp computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  unsigned dynamicCastCount = 0;
  computeFn.walk([&](Operation *op) {
    bool isDynamicNonAssertCast =
        (isa<cast::IntToFeltOp>(op) &&
         llvm::cast<cast::IntToFeltOp>(op).getOverflow() != cast::OverflowSemantics::ASSERT) ||
        (isa<cast::FeltToIndexOp>(op) &&
         llvm::cast<cast::FeltToIndexOp>(op).getOverflow() != cast::OverflowSemantics::ASSERT);
    if (!isDynamicNonAssertCast) {
      return;
    }

    ++dynamicCastCount;
    const IntervalAnalysisLattice *lattice = lookupLattice(analysis, op->getResult(0));
    ASSERT_NE(lattice, nullptr);
    const ExpressionValue &value = lattice->getValue().getScalarValue();
    EXPECT_NE(value.getExpr(), nullptr);
    EXPECT_EQ(value.getInterval(), Interval::Entire(field));
    EXPECT_FALSE(value.hasUnreducedInterval());
  });
  EXPECT_EQ(dynamicCastCount, 6U);
}

// Truncation reduces its low-bit result into [0, p). Subsequent unreduced
// arithmetic and stored member summaries must start from that canonical range.
TEST_F(IntervalAnalysisAPITests, DynamicTruncationReducesToFieldRange) {
  auto mod = parseModule(kDynamicNonAssertCastModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  unsigned toFeltCount = 0;
  computeFn.walk([&](cast::IntToFeltOp op) {
    ++toFeltCount;
    SCOPED_TRACE(cast::stringifyOverflowSemantics(op.getOverflow()).str());
    const auto *lattice = lookupLattice(analysis, op.getResult());
    ASSERT_NE(lattice, nullptr);
    const auto &value = lattice->getValue().getScalarValue();
    EXPECT_NE(value.getExpr(), nullptr);
    EXPECT_EQ(value.getInterval(), Interval::Entire(field));
    ASSERT_TRUE(value.hasUnreducedInterval());
    AssertUnreducedIntervalEq(
        UnreducedInterval(field.zero(), field.maxVal()), value.getUnreducedInterval()
    );
  });
  EXPECT_EQ(toFeltCount, 3U);

  felt::AddFeltOp sumOp;
  computeFn.walk([&](felt::AddFeltOp op) { sumOp = op; });
  ASSERT_TRUE(sumOp != nullptr);
  const auto *sumLattice = lookupLattice(analysis, sumOp.getResult());
  ASSERT_NE(sumLattice, nullptr);
  const auto &sumValue = sumLattice->getValue().getScalarValue();
  EXPECT_EQ(sumValue.getInterval(), Interval::Entire(field));
  ASSERT_TRUE(sumValue.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.one(), field.prime()), sumValue.getUnreducedInterval()
  );

  const auto &intervals = analysis.getResult(structDef).getComputeUnreducedIntervals();
  unsigned memberCount = 0;
  for (auto member : structDef.getOps<MemberDefOp>()) {
    ++memberCount;
    SCOPED_TRACE(member.getName().str());
    SourceRef ref(
        llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(member)}
    );
    const auto *it = intervals.find(ref);
    ASSERT_NE(it, intervals.end());
    auto expected = member.getName() == "truncated"
                        ? UnreducedInterval(field.zero(), field.maxVal())
                        : UnreducedInterval(field.one(), field.prime());
    AssertUnreducedIntervalEq(expected, it->second);
  }
  EXPECT_EQ(memberCount, 2U);
}

TEST_F(IntervalAnalysisAPITests, ConstrainIntervalsFindMatchesStoredArrayRefs) {
  auto mod = parseModule(kArrayIntervalModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  ASSERT_TRUE(constrainFn != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setPropagateInputConstraints(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const auto &intervals = analysis.getResult(structDef).getConstrainIntervals();
  ASSERT_FALSE(intervals.empty());

  // Iteration and lookup should agree for every stored key.
  for (const auto &[ref, interval] : intervals) {
    const auto *it = intervals.find(ref);
    ASSERT_NE(it, intervals.end()) << "missing key on self-lookup: " << buildStringViaPrint(ref);
    ASSERT_TRUE(checkCond(interval, it->second, interval == it->second))
        << buildStringViaPrint(ref);
  }

  MemberDefOp outMember;
  for (auto member : structDef.getOps<MemberDefOp>()) {
    if (member.getName() == "out") {
      outMember = member;
      break;
    }
  }
  ASSERT_TRUE(outMember != nullptr);

  SourceRef outRef(constrainFn.getArgument(0), {SourceRefIndex(outMember)});
  for (int64_t i = 0; i < 3; i++) {
    auto elemRef = outRef.createChild(SourceRefIndex(i));
    ASSERT_TRUE(succeeded(elemRef));
    const auto *it = intervals.find(*elemRef);
    ASSERT_NE(it, intervals.end())
        << "missing constrain interval for " << buildStringViaPrint(*elemRef);
    ASSERT_TRUE(it->second.isDegenerate())
        << buildStringViaPrint(*elemRef) << " -> " << buildStringViaPrint(it->second);
    ASSERT_EQ(it->second.lhs(), field.felt(i + 1)) << buildStringViaPrint(*elemRef);
  }
}

TEST_F(IntervalAnalysisAPITests, ComputeIntervalsTrackArrayNewStoredIntoMember) {
  auto mod = parseModule(kComputeArrayMemberWriteModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const auto &intervals = analysis.getResult(structDef).getComputeIntervals();
  ASSERT_FALSE(intervals.empty());

  MemberDefOp outMember;
  for (auto member : structDef.getOps<MemberDefOp>()) {
    if (member.getName() == "out") {
      outMember = member;
      break;
    }
  }
  ASSERT_TRUE(outMember != nullptr);

  SourceRef outRef(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(outMember)}
  );
  auto out0Ref = outRef.createChild(SourceRefIndex(0));
  auto out1Ref = outRef.createChild(SourceRefIndex(1));
  ASSERT_TRUE(succeeded(out0Ref));
  ASSERT_TRUE(succeeded(out1Ref));

  const auto *out0It = intervals.find(*out0Ref);
  ASSERT_NE(out0It, intervals.end())
      << "missing compute interval for " << buildStringViaPrint(*out0Ref);
  ASSERT_TRUE(out0It->second.isDegenerate())
      << buildStringViaPrint(*out0Ref) << " -> " << buildStringViaPrint(out0It->second);
  ASSERT_EQ(out0It->second.lhs(), field.zero()) << buildStringViaPrint(*out0Ref);

  const auto *out1It = intervals.find(*out1Ref);
  ASSERT_NE(out1It, intervals.end())
      << "missing compute interval for " << buildStringViaPrint(*out1Ref);
  auto expected = Interval::TypeA(field, field.felt(5), field.felt(6));
  ASSERT_TRUE(checkCond(expected, out1It->second, expected == out1It->second))
      << buildStringViaPrint(*out1Ref) << " -> " << buildStringViaPrint(out1It->second);
}

TEST_F(IntervalAnalysisAPITests, ScfWhilePreservesCarriedPodArraySourceRefs) {
  auto mod = parseModule(kScfCarriedPodArrayModule);
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  scf::WhileOp whileOp;
  array::ReadArrayOp resultRead;
  pod::ReadPodOp resultPodRead;
  computeFn.walk([&](scf::WhileOp candidate) { whileOp = candidate; });
  ASSERT_TRUE(whileOp != nullptr);
  computeFn.walk([&](array::ReadArrayOp candidateArrayRead) {
    if (candidateArrayRead.getArrRef() == whileOp.getResult(1)) {
      resultRead = candidateArrayRead;
    }
  });
  ASSERT_TRUE(resultRead != nullptr);
  computeFn.walk([&](pod::ReadPodOp candidatePodRead) {
    if (candidatePodRead.getPodRef() == resultRead.getResult()) {
      resultPodRead = candidatePodRead;
    }
  });
  ASSERT_TRUE(resultPodRead != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  SourceRefLatticeValue carried =
      SourceRefAnalysis::getValueState(analysis.getSolver(), whileOp.getResult(1));
  ASSERT_TRUE(carried.isArray());
  ASSERT_EQ(carried.getArraySize(), 2U);

  SourceRefLatticeValue readState =
      SourceRefAnalysis::getValueState(analysis.getSolver(), resultRead.getResult());
  ASSERT_TRUE(readState.isSingleValue());
  SourceRefLatticeValue scalarState =
      SourceRefAnalysis::getValueState(analysis.getSolver(), resultPodRead.getResult());
  ASSERT_TRUE(scalarState.isSingleValue());

  const IntervalAnalysisLattice *resultLattice = lookupLattice(analysis, resultPodRead.getResult());
  ASSERT_NE(resultLattice, nullptr);
  const Interval &resultInterval = resultLattice->getValue().getScalarValue().getInterval();
  // Loop-carried values are deliberately widened, but the post-loop query must remain usable.
  EXPECT_TRUE(resultInterval.isEntire());
}

TEST_F(IntervalAnalysisAPITests, UnreducedIntervalsDisabledByDefault) {
  auto mod = parseModule(kUnreducedIntervalPropagationModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  felt::FeltConstantOp constFive;
  felt::AddFeltOp sumOp;
  felt::NegFeltOp negOp;
  felt::UnsignedIntDivFeltOp divOp;
  computeFn.walk([&](Operation *op) {
    if (auto c = dyn_cast<felt::FeltConstantOp>(op)) {
      constFive = c;
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      sumOp = add;
    } else if (auto neg = dyn_cast<felt::NegFeltOp>(op)) {
      negOp = neg;
    } else if (auto div = dyn_cast<felt::UnsignedIntDivFeltOp>(op)) {
      divOp = div;
    }
  });
  ASSERT_TRUE(constFive != nullptr);
  ASSERT_TRUE(sumOp != nullptr);
  ASSERT_TRUE(negOp != nullptr);
  ASSERT_TRUE(divOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  SmallVector<Value> values = {
      computeFn.getArgument(0), constFive.getResult(), sumOp.getResult(), negOp.getResult(),
      divOp.getResult()
  };
  for (Value value : values) {
    const IntervalAnalysisLattice *lattice = lookupLattice(analysis, value);
    ASSERT_NE(lattice, nullptr);
    EXPECT_FALSE(lattice->getValue().getScalarValue().hasUnreducedInterval())
        << buildStringViaPrint(value);
  }
}

TEST_F(IntervalAnalysisAPITests, UnreducedIntervalsPropagateThroughSupportedArithmetic) {
  auto mod = parseModule(kUnreducedIntervalPropagationModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  felt::FeltConstantOp constFive;
  felt::AddFeltOp sumOp;
  felt::NegFeltOp negOp;
  felt::UnsignedIntDivFeltOp divOp;
  computeFn.walk([&](Operation *op) {
    if (auto c = dyn_cast<felt::FeltConstantOp>(op)) {
      constFive = c;
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      sumOp = add;
    } else if (auto neg = dyn_cast<felt::NegFeltOp>(op)) {
      negOp = neg;
    } else if (auto div = dyn_cast<felt::UnsignedIntDivFeltOp>(op)) {
      divOp = div;
    }
  });
  ASSERT_TRUE(constFive != nullptr);
  ASSERT_TRUE(sumOp != nullptr);
  ASSERT_TRUE(negOp != nullptr);
  ASSERT_TRUE(divOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *argLattice = lookupLattice(analysis, computeFn.getArgument(0));
  ASSERT_NE(argLattice, nullptr);
  const ExpressionValue &argExpr = argLattice->getValue().getScalarValue();
  ASSERT_TRUE(argExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.zero(), field.maxVal()), argExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *constLattice = lookupLattice(analysis, constFive.getResult());
  ASSERT_NE(constLattice, nullptr);
  const ExpressionValue &constExpr = constLattice->getValue().getScalarValue();
  ASSERT_TRUE(constExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(5), field.felt(5)), constExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *sumLattice = lookupLattice(analysis, sumOp.getResult());
  ASSERT_NE(sumLattice, nullptr);
  const ExpressionValue &sumExpr = sumLattice->getValue().getScalarValue();
  ASSERT_TRUE(sumExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(5), field.maxVal() + field.felt(5)),
      sumExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *negLattice = lookupLattice(analysis, negOp.getResult());
  ASSERT_NE(negLattice, nullptr);
  const ExpressionValue &negExpr = negLattice->getValue().getScalarValue();
  ASSERT_TRUE(negExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(-(field.maxVal() + field.felt(5)), -field.felt(5)),
      negExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *divLattice = lookupLattice(analysis, divOp.getResult());
  ASSERT_NE(divLattice, nullptr);
  EXPECT_FALSE(divLattice->getValue().getScalarValue().hasUnreducedInterval());
}

TEST_F(IntervalAnalysisAPITests, UnreducedIntervalsTrackBooleanAndSelectResults) {
  auto mod = parseModule(kUnreducedBoolAndSelectModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  arith::ConstantOp trueConst;
  arith::ConstantOp falseConst;
  boolean::XorBoolOp xorOp;
  boolean::NotBoolOp notOp;
  SmallVector<arith::SelectOp> selectOps;
  computeFn.walk([&](Operation *op) {
    if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
      if (auto boolAttr = dyn_cast<BoolAttr>(cst.getValue())) {
        if (boolAttr.getValue()) {
          trueConst = cst;
        } else {
          falseConst = cst;
        }
      }
    } else if (auto xorBool = dyn_cast<boolean::XorBoolOp>(op)) {
      xorOp = xorBool;
    } else if (auto notBool = dyn_cast<boolean::NotBoolOp>(op)) {
      notOp = notBool;
    } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
      selectOps.push_back(select);
    }
  });
  ASSERT_TRUE(trueConst != nullptr);
  ASSERT_TRUE(falseConst != nullptr);
  ASSERT_TRUE(xorOp != nullptr);
  ASSERT_TRUE(notOp != nullptr);
  ASSERT_EQ(selectOps.size(), 3U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *flagLattice = lookupLattice(analysis, computeFn.getArgument(0));
  ASSERT_NE(flagLattice, nullptr);
  const ExpressionValue &flagExpr = flagLattice->getValue().getScalarValue();
  ASSERT_TRUE(flagExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(UnreducedInterval(0, 1), flagExpr.getUnreducedInterval());

  const IntervalAnalysisLattice *trueLattice = lookupLattice(analysis, trueConst.getResult());
  ASSERT_NE(trueLattice, nullptr);
  ASSERT_TRUE(trueLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(1, 1), trueLattice->getValue().getScalarValue().getUnreducedInterval()
  );

  const IntervalAnalysisLattice *falseLattice = lookupLattice(analysis, falseConst.getResult());
  ASSERT_NE(falseLattice, nullptr);
  ASSERT_TRUE(falseLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(0, 0), falseLattice->getValue().getScalarValue().getUnreducedInterval()
  );

  const IntervalAnalysisLattice *xorLattice = lookupLattice(analysis, xorOp.getResult());
  ASSERT_NE(xorLattice, nullptr);
  ASSERT_TRUE(xorLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(1, 1), xorLattice->getValue().getScalarValue().getUnreducedInterval()
  );

  const IntervalAnalysisLattice *notLattice = lookupLattice(analysis, notOp.getResult());
  ASSERT_NE(notLattice, nullptr);
  ASSERT_TRUE(notLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(0, 1), notLattice->getValue().getScalarValue().getUnreducedInterval()
  );

  const IntervalAnalysisLattice *selectTrueLattice =
      lookupLattice(analysis, selectOps[0].getResult());
  ASSERT_NE(selectTrueLattice, nullptr);
  ASSERT_TRUE(selectTrueLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(3), field.felt(3)),
      selectTrueLattice->getValue().getScalarValue().getUnreducedInterval()
  );

  const IntervalAnalysisLattice *selectFalseLattice =
      lookupLattice(analysis, selectOps[1].getResult());
  ASSERT_NE(selectFalseLattice, nullptr);
  ASSERT_TRUE(selectFalseLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(9), field.felt(9)),
      selectFalseLattice->getValue().getScalarValue().getUnreducedInterval()
  );

  const IntervalAnalysisLattice *selectFlagLattice =
      lookupLattice(analysis, selectOps[2].getResult());
  ASSERT_NE(selectFlagLattice, nullptr);
  ASSERT_TRUE(selectFlagLattice->getValue().getScalarValue().hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(3), field.felt(9)),
      selectFlagLattice->getValue().getScalarValue().getUnreducedInterval()
  );
}

TEST_F(IntervalAnalysisAPITests, RefinedReducedIntervalsDropUnreducedIntervals) {
  auto mod = parseModule(kUnreducedTypeFLiftModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  felt::NegFeltOp negOp;
  computeFn.walk([&](felt::NegFeltOp op) { negOp = op; });
  ASSERT_TRUE(negOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *argLattice = lookupLattice(analysis, computeFn.getArgument(0));
  ASSERT_NE(argLattice, nullptr);
  EXPECT_FALSE(argLattice->getValue().getScalarValue().hasUnreducedInterval());

  const IntervalAnalysisLattice *negLattice = lookupLattice(analysis, negOp.getResult());
  ASSERT_NE(negLattice, nullptr);
  EXPECT_FALSE(negLattice->getValue().getScalarValue().hasUnreducedInterval());
}

TEST_F(IntervalAnalysisAPITests, TypeFConstraintPropagationUsesFirstUnreducedInterval) {
  auto mod = parseModule(kUnreducedTypeFPropagationModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setPropagateInputConstraints(true);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *argLattice = lookupLattice(analysis, computeFn.getArgument(0));
  ASSERT_NE(argLattice, nullptr);
  const ExpressionValue &argExpr = argLattice->getValue().getScalarValue();
  ASSERT_TRUE(argExpr.getInterval().isTypeF());
  ASSERT_TRUE(argExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(UnreducedInterval(-1, 0), argExpr.getUnreducedInterval());
}

TEST_F(IntervalAnalysisAPITests, UnconstrainedSignalReadsDefaultToCanonicalUnreducedInterval) {
  auto mod = parseModule(kUnreducedSignalReadDefaultModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  ASSERT_TRUE(constrainFn != nullptr);

  component::MemberReadOp readOp;
  constrainFn.walk([&](component::MemberReadOp op) { readOp = op; });
  ASSERT_TRUE(readOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *readLattice = lookupLattice(analysis, readOp.getResult());
  ASSERT_NE(readLattice, nullptr);
  const ExpressionValue &readExpr = readLattice->getValue().getScalarValue();
  ASSERT_TRUE(readExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.zero(), field.maxVal()), readExpr.getUnreducedInterval()
  );
}

TEST_F(IntervalAnalysisAPITests, IsZeroTracksWideUnreducedMulAndSumIntervals) {
  auto mod = parseModule(kUnreducedIsZeroModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  ASSERT_TRUE(computeFn != nullptr);
  ASSERT_TRUE(constrainFn != nullptr);

  llvm::SmallVector<felt::InvFeltOp> computeInvOps;
  llvm::SmallVector<felt::NegFeltOp> computeNegOps;
  llvm::SmallVector<felt::MulFeltOp> computeMulOps;
  llvm::SmallVector<felt::AddFeltOp> computeAddOps;
  computeFn.walk([&](Operation *op) {
    if (auto inv = dyn_cast<felt::InvFeltOp>(op)) {
      computeInvOps.push_back(inv);
    } else if (auto neg = dyn_cast<felt::NegFeltOp>(op)) {
      computeNegOps.push_back(neg);
    } else if (auto mul = dyn_cast<felt::MulFeltOp>(op)) {
      computeMulOps.push_back(mul);
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      computeAddOps.push_back(add);
    }
  });
  ASSERT_EQ(computeInvOps.size(), 1U);
  ASSERT_EQ(computeNegOps.size(), 1U);
  ASSERT_EQ(computeMulOps.size(), 1U);
  ASSERT_EQ(computeAddOps.size(), 1U);

  llvm::SmallVector<felt::NegFeltOp> constrainNegOps;
  llvm::SmallVector<felt::MulFeltOp> constrainMulOps;
  llvm::SmallVector<felt::AddFeltOp> constrainAddOps;
  constrainFn.walk([&](Operation *op) {
    if (auto neg = dyn_cast<felt::NegFeltOp>(op)) {
      constrainNegOps.push_back(neg);
    } else if (auto mul = dyn_cast<felt::MulFeltOp>(op)) {
      constrainMulOps.push_back(mul);
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      constrainAddOps.push_back(add);
    }
  });
  ASSERT_EQ(constrainNegOps.size(), 1U);
  ASSERT_EQ(constrainMulOps.size(), 2U);
  ASSERT_EQ(constrainAddOps.size(), 1U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  auto pMinusOne = field.maxVal();
  auto square = pMinusOne * pMinusOne;
  auto oneMinusSquare = field.one() - square;

  auto assertValueHasUnreduced = [&](Value value, const UnreducedInterval &expected) {
    const IntervalAnalysisLattice *lattice = lookupLattice(analysis, value);
    ASSERT_NE(lattice, nullptr);
    const ExpressionValue &expr = lattice->getValue().getScalarValue();
    ASSERT_TRUE(expr.hasUnreducedInterval());
    AssertUnreducedIntervalEq(expected, expr.getUnreducedInterval());
  };

  assertValueHasUnreduced(
      computeInvOps.front().getResult(), UnreducedInterval(field.zero(), field.maxVal())
  );
  assertValueHasUnreduced(
      computeNegOps.front().getResult(), UnreducedInterval(-pMinusOne, field.zero())
  );
  assertValueHasUnreduced(
      computeMulOps.front().getResult(), UnreducedInterval(-square, field.zero())
  );
  assertValueHasUnreduced(
      computeAddOps.front().getResult(), UnreducedInterval(oneMinusSquare, field.one())
  );

  assertValueHasUnreduced(
      constrainNegOps.front().getResult(), UnreducedInterval(-pMinusOne, field.zero())
  );
  assertValueHasUnreduced(
      constrainMulOps.front().getResult(), UnreducedInterval(-square, field.zero())
  );
  assertValueHasUnreduced(
      constrainAddOps.front().getResult(), UnreducedInterval(oneMinusSquare, field.one())
  );
}

TEST_F(IntervalAnalysisAPITests, ProductFunctionsTrackReducedIntervals) {
  auto mod = parseModule(kProductFunctionIntervalModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto productFn = structDef.getProductFuncOp();
  ASSERT_TRUE(productFn != nullptr);

  felt::AddFeltOp sumOp;
  component::MemberReadOp readOp;
  productFn.walk([&](Operation *op) {
    if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      sumOp = add;
    } else if (auto read = dyn_cast<component::MemberReadOp>(op)) {
      readOp = read;
    }
  });
  ASSERT_TRUE(sumOp != nullptr);
  ASSERT_TRUE(readOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *argLattice = lookupLattice(analysis, productFn.getArgument(0));
  ASSERT_NE(argLattice, nullptr);
  const ExpressionValue &argExpr = argLattice->getValue().getScalarValue();
  ASSERT_TRUE(checkCond(
      Interval::TypeA(field, field.felt(5), field.felt(6)), argExpr.getInterval(),
      argExpr.getInterval() == Interval::TypeA(field, field.felt(5), field.felt(6))
  ));

  const IntervalAnalysisLattice *sumLattice = lookupLattice(analysis, sumOp.getResult());
  ASSERT_NE(sumLattice, nullptr);
  const ExpressionValue &sumExpr = sumLattice->getValue().getScalarValue();
  ASSERT_TRUE(checkCond(
      Interval::TypeA(field, field.felt(10), field.felt(11)), sumExpr.getInterval(),
      sumExpr.getInterval() == Interval::TypeA(field, field.felt(10), field.felt(11))
  ));

  const IntervalAnalysisLattice *readLattice = lookupLattice(analysis, readOp.getResult());
  ASSERT_NE(readLattice, nullptr);
  const ExpressionValue &readExpr = readLattice->getValue().getScalarValue();
  ASSERT_TRUE(checkCond(
      Interval::TypeA(field, field.felt(10), field.felt(11)), readExpr.getInterval(),
      readExpr.getInterval() == Interval::TypeA(field, field.felt(10), field.felt(11))
  ));
}

TEST_F(IntervalAnalysisAPITests, ProductFunctionsTrackUnreducedIntervals) {
  auto mod = parseModule(kProductFunctionUnreducedModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto productFn = structDef.getProductFuncOp();
  ASSERT_TRUE(productFn != nullptr);

  felt::FeltConstantOp constFive;
  felt::AddFeltOp sumOp;
  component::MemberReadOp readOp;
  productFn.walk([&](Operation *op) {
    if (auto cst = dyn_cast<felt::FeltConstantOp>(op)) {
      constFive = cst;
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      sumOp = add;
    } else if (auto read = dyn_cast<component::MemberReadOp>(op)) {
      readOp = read;
    }
  });
  ASSERT_TRUE(constFive != nullptr);
  ASSERT_TRUE(sumOp != nullptr);
  ASSERT_TRUE(readOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.setTrackUnreducedIntervals(true);
  ASSERT_TRUE(succeeded(analysis.runAnalysis(am)));

  const IntervalAnalysisLattice *argLattice = lookupLattice(analysis, productFn.getArgument(0));
  ASSERT_NE(argLattice, nullptr);
  const ExpressionValue &argExpr = argLattice->getValue().getScalarValue();
  ASSERT_TRUE(argExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.zero(), field.maxVal()), argExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *constLattice = lookupLattice(analysis, constFive.getResult());
  ASSERT_NE(constLattice, nullptr);
  const ExpressionValue &constExpr = constLattice->getValue().getScalarValue();
  ASSERT_TRUE(constExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(5), field.felt(5)), constExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *sumLattice = lookupLattice(analysis, sumOp.getResult());
  ASSERT_NE(sumLattice, nullptr);
  const ExpressionValue &sumExpr = sumLattice->getValue().getScalarValue();
  ASSERT_TRUE(sumExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(5), field.maxVal() + field.felt(5)),
      sumExpr.getUnreducedInterval()
  );

  const IntervalAnalysisLattice *readLattice = lookupLattice(analysis, readOp.getResult());
  ASSERT_NE(readLattice, nullptr);
  const ExpressionValue &readExpr = readLattice->getValue().getScalarValue();
  ASSERT_TRUE(readExpr.hasUnreducedInterval());
  AssertUnreducedIntervalEq(
      UnreducedInterval(field.felt(5), field.maxVal() + field.felt(5)),
      readExpr.getUnreducedInterval()
  );
}

// Equal intervals must hash identically even when their endpoints use different
// DynamicAPInt representations (native integers versus 256-bit APInts). Cover
// both unreduced and field intervals to preserve the equality/hash contract.
TEST_F(IntervalTests, HashIgnoresIntegerRepresentation) {
  DynamicAPInt narrow(7), wide(llvm::APInt(256, 7));
  UnreducedInterval a(narrow, narrow + 1), b(wide, wide + 1);
  EXPECT_EQ(a, b);
  EXPECT_EQ(UnreducedInterval::Hash {}(a), UnreducedInterval::Hash {}(b));
  auto x = Interval::Degenerate(f, narrow);
  auto y = Interval::Degenerate(f, wide);
  EXPECT_EQ(x, y);
  EXPECT_EQ(Interval::Hash {}(x), Interval::Hash {}(y));
}
