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
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"

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

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
    EXPECT_TRUE(mod);
    return mod;
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
