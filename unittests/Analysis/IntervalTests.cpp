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

TEST_F(IntervalTests, BoolXor) {
  auto falseInterval = Interval::False(f);
  auto trueInterval = Interval::True(f);
  auto boolInterval = Interval::Boolean(f);

  AssertIntervalEq(falseInterval, boolXor(trueInterval, trueInterval));
  AssertIntervalEq(trueInterval, boolXor(trueInterval, falseInterval));
  AssertIntervalEq(boolInterval, boolXor(boolInterval, trueInterval));
}

TEST_F(IntervalTests, Mod) {
  AssertIntervalEq(interval(0, 8), entire % degen(8));
  AssertIntervalEq(interval(0, 10), interval(0, 100) % interval(1, 10));
  AssertIntervalEq(interval(0, 1000), degen(7) % interval(0, 1000));
  AssertIntervalEq(degen(0), empty % empty);
  AssertIntervalEq(degen(0), entire % empty);
  AssertIntervalEq(degen(0), empty % entire);
  AssertIntervalEq(degen(0), degen(1) % entire);
  // any % typeF == entire
  auto typeF = UnreducedInterval(f.half() + f.one(), f.prime() + f.one()).reduce(f);
  ASSERT_TRUE(typeF.isTypeF());
  AssertIntervalEq(interval(0, 1), interval(7, 8) % typeF);
}

class IntervalAnalysisAPITests : public LLZKTest {
protected:
  inline static void
  AssertUnreducedIntervalEq(const UnreducedInterval &expected, const UnreducedInterval &actual) {
    ASSERT_TRUE(checkCond(expected, actual, expected == actual));
  }

  static constexpr auto kUnreducedIntervalPropagationModule = R"mlir(
module attributes {veridise.lang = "llzk"} {
  struct.def @TrackUnreduced {
    struct.field @out : !felt.type {llzk.pub, signal}

    function.def @compute(%a: !felt.type) -> !struct.type<@TrackUnreduced>
        attributes {function.allow_witness, function.allow_non_native_field_ops} {
      %self = struct.new : <@TrackUnreduced>
      %felt_const_5 = felt.const 5
      %sum = felt.add %a, %felt_const_5 : !felt.type, !felt.type
      %neg = felt.neg %sum : !felt.type
      struct.writef %self[@out] = %neg : <@TrackUnreduced>, !felt.type
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
module attributes {veridise.lang = "llzk"} {
  struct.def @TrackUnreducedBoolAndSelect {
    struct.field @out : !felt.type {llzk.pub, signal}

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
      struct.writef %self[@out] = %sel_flag : <@TrackUnreducedBoolAndSelect>, !felt.type
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
module attributes {veridise.lang = "llzk"} {
  struct.def @TrackUnreducedTypeF {
    struct.field @out : !felt.type {llzk.pub, signal}

    function.def @compute(%a: !felt.type) -> !struct.type<@TrackUnreducedTypeF>
        attributes {function.allow_witness, function.allow_non_native_field_ops} {
      %self = struct.new : <@TrackUnreducedTypeF>
      %felt_const_1 = felt.const 1
      %cmp = bool.cmp le(%a, %felt_const_1)
      bool.assert %cmp
      %neg = felt.neg %a : !felt.type
      struct.writef %self[@out] = %neg : <@TrackUnreducedTypeF>, !felt.type
      function.return %self : !struct.type<@TrackUnreducedTypeF>
    }

    function.def @constrain(%self: !struct.type<@TrackUnreducedTypeF>, %a: !felt.type)
        attributes {function.allow_constraint} {
      %read = struct.readf %self[@out] : <@TrackUnreducedTypeF>, !felt.type
      %felt_const_0 = felt.const 0
      %sum = felt.add %read, %felt_const_0 : !felt.type, !felt.type
      function.return
    }
  }
}
)mlir";

  static constexpr auto kUnreducedTypeFPropagationModule = R"mlir(
module attributes {veridise.lang = "llzk"} {
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
      %cmp = bool.cmp le(%sum, %felt_const_1)
      bool.assert %cmp
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

TEST_F(IntervalAnalysisAPITests, UnreducedIntervalsDisabledByDefault) {
  auto mod = parseModule(kUnreducedIntervalPropagationModule);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  ASSERT_TRUE(computeFn != nullptr);

  felt::FeltConstantOp constFive;
  felt::AddFeltOp sumOp;
  felt::NegFeltOp negOp;
  computeFn.walk([&](Operation *op) {
    if (auto c = dyn_cast<felt::FeltConstantOp>(op)) {
      constFive = c;
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      sumOp = add;
    } else if (auto neg = dyn_cast<felt::NegFeltOp>(op)) {
      negOp = neg;
    }
  });
  ASSERT_TRUE(constFive != nullptr);
  ASSERT_TRUE(sumOp != nullptr);
  ASSERT_TRUE(negOp != nullptr);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ModuleIntervalAnalysis analysis(mod->getOperation());
  const Field &field = Field::getField("babybear");
  analysis.setField(field);
  analysis.runAnalysis(am);

  SmallVector<Value> values = {
      computeFn.getArgument(0), constFive.getResult(), sumOp.getResult(), negOp.getResult()
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
  computeFn.walk([&](Operation *op) {
    if (auto c = dyn_cast<felt::FeltConstantOp>(op)) {
      constFive = c;
    } else if (auto add = dyn_cast<felt::AddFeltOp>(op)) {
      sumOp = add;
    } else if (auto neg = dyn_cast<felt::NegFeltOp>(op)) {
      negOp = neg;
    }
  });
  ASSERT_TRUE(constFive != nullptr);
  ASSERT_TRUE(sumOp != nullptr);
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
