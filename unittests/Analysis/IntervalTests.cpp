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

#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Analysis/Intervals.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>
#include <string>

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

TEST_F(IntervalTests, UnreducedIntervalOverlap) {
  UnreducedInterval a(0, 100), b(100, 200), c(101, 300), d(1, 0);
  ASSERT_TRUE(a.overlaps(b));
  ASSERT_TRUE(b.overlaps(a));
  ASSERT_FALSE(a.overlaps(c));
  ASSERT_TRUE(b.overlaps(c));
  ASSERT_FALSE(d.overlaps(a));
}

TEST_F(IntervalTests, SourceRefIndexHalfOpenOverlap) {
  SourceRefIndex range(APInt(64, 2), APInt(64, 5));
  SourceRefIndex overlappingRange(APInt(64, 4), APInt(64, 7));
  SourceRefIndex adjacentRange(APInt(64, 5), APInt(64, 8));

  EXPECT_FALSE(range.overlaps(SourceRefIndex(1)));
  EXPECT_TRUE(range.overlaps(SourceRefIndex(2)));
  EXPECT_TRUE(range.overlaps(SourceRefIndex(4)));
  EXPECT_FALSE(range.overlaps(SourceRefIndex(5)));
  EXPECT_TRUE(range.overlaps(overlappingRange));
  EXPECT_FALSE(range.overlaps(adjacentRange));
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

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
    EXPECT_TRUE(mod);
    return mod;
  }

  const IntervalAnalysisLattice *lookupLattice(ModuleIntervalAnalysis &analysis, Value value) {
    return analysis.getSolver().lookupState<IntervalAnalysisLattice>(value);
  }
};

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(outMember)}
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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
  analysis.runAnalysis(am);

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
