//===-- BoolFoldTests.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Bool/IR/Enums.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::boolean;

//===------------------------------------------------------------------===//
// Test infrastructure
//===------------------------------------------------------------------===//

class BoolFoldTest : public LLZKTest {
protected:
  /// bitwidth large enough to hold babybear prime (which is 31 bits) with optional sign bit.
  static constexpr unsigned BITWIDTH = 32;
  /// babybear: p = 2013265921, half = (p+1)/2 = 1006632961
  static constexpr uint64_t BB_PRIME = 2013265921ULL;
  static constexpr StringLiteral BB_FIELD = "babybear";

  /// Build a 1-bit integer attribute (i1).
  IntegerAttr boolAttr(bool val) {
    return IntegerAttr::get(IntegerType::get(&ctx, 1), val ? 1 : 0);
  }

  FeltConstAttr babyBearConst(uint64_t val) {
    return FeltConstAttr::get(&ctx, APInt(BITWIDTH, val), BB_FIELD);
  }

  /// Insert two arith.constant i1 ops into a detached block, build a binary
  /// Bool op on them, call fold with the constant attributes, and return the
  /// folded IntegerAttr (or null if no fold occurred).
  template <typename OpTy> IntegerAttr foldBoolBinary(IntegerAttr lhsAttr, IntegerAttr rhsAttr) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    auto lhsOp = builder.create<arith::ConstantOp>(loc, lhsAttr);
    auto rhsOp = builder.create<arith::ConstantOp>(loc, rhsAttr);
    auto op = builder.create<OpTy>(loc, lhsOp.getResult(), rhsOp.getResult());

    SmallVector<Attribute, 2> operands = {lhsAttr, rhsAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(op->fold(operands, results)) || results.empty()) {
      return {};
    }
    return dyn_cast_or_null<IntegerAttr>(dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Same as foldBoolBinary but for unary Bool ops.
  template <typename OpTy> IntegerAttr foldBoolUnary(IntegerAttr operandAttr) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    auto operandOp = builder.create<arith::ConstantOp>(loc, operandAttr);
    auto op = builder.create<OpTy>(loc, operandOp.getResult());

    SmallVector<Attribute, 1> operands = {operandAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(op->fold(operands, results)) || results.empty()) {
      return {};
    }
    return dyn_cast_or_null<IntegerAttr>(dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Build a CmpOp from two FeltConstAttr values with the given predicate,
  /// call fold, and return the folded i1 IntegerAttr (or null if no fold).
  IntegerAttr foldCmp(FeltCmpPredicate pred, FeltConstAttr lhsAttr, FeltConstAttr rhsAttr) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    auto lhsOp = builder.create<FeltConstantOp>(loc, lhsAttr);
    auto rhsOp = builder.create<FeltConstantOp>(loc, rhsAttr);
    auto predAttr = FeltCmpPredicateAttr::get(&ctx, pred);
    auto cmpOp = builder.create<CmpOp>(loc, predAttr, lhsOp.getResult(), rhsOp.getResult());

    SmallVector<Attribute, 2> operands = {lhsAttr, rhsAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(cmpOp->fold(operands, results)) || results.empty()) {
      return {};
    }
    return dyn_cast_or_null<IntegerAttr>(dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Assert that fold produced the expected boolean value.
  void expectBool(IntegerAttr result, bool expected) {
    ASSERT_TRUE(result) << "expected fold to succeed";
    EXPECT_EQ(result.getValue().getBoolValue(), expected);
  }

  /// Assert that no fold occurred.
  void expectNoFold(IntegerAttr result) { EXPECT_FALSE(result) << "expected fold to be skipped"; }
};

//===------------------------------------------------------------------===//
// bool.and
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, AndTrueTrue) {
  expectBool(foldBoolBinary<AndBoolOp>(boolAttr(true), boolAttr(true)), true);
}

TEST_F(BoolFoldTest, AndTrueFalse) {
  expectBool(foldBoolBinary<AndBoolOp>(boolAttr(true), boolAttr(false)), false);
}

TEST_F(BoolFoldTest, AndFalseTrue) {
  expectBool(foldBoolBinary<AndBoolOp>(boolAttr(false), boolAttr(true)), false);
}

TEST_F(BoolFoldTest, AndFalseFalse) {
  expectBool(foldBoolBinary<AndBoolOp>(boolAttr(false), boolAttr(false)), false);
}

//===------------------------------------------------------------------===//
// bool.or
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, OrTrueTrue) {
  expectBool(foldBoolBinary<OrBoolOp>(boolAttr(true), boolAttr(true)), true);
}

TEST_F(BoolFoldTest, OrTrueFalse) {
  expectBool(foldBoolBinary<OrBoolOp>(boolAttr(true), boolAttr(false)), true);
}

TEST_F(BoolFoldTest, OrFalseTrue) {
  expectBool(foldBoolBinary<OrBoolOp>(boolAttr(false), boolAttr(true)), true);
}

TEST_F(BoolFoldTest, OrFalseFalse) {
  expectBool(foldBoolBinary<OrBoolOp>(boolAttr(false), boolAttr(false)), false);
}

//===------------------------------------------------------------------===//
// bool.xor
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, XorTrueTrue) {
  expectBool(foldBoolBinary<XorBoolOp>(boolAttr(true), boolAttr(true)), false);
}

TEST_F(BoolFoldTest, XorTrueFalse) {
  expectBool(foldBoolBinary<XorBoolOp>(boolAttr(true), boolAttr(false)), true);
}

TEST_F(BoolFoldTest, XorFalseTrue) {
  expectBool(foldBoolBinary<XorBoolOp>(boolAttr(false), boolAttr(true)), true);
}

TEST_F(BoolFoldTest, XorFalseFalse) {
  expectBool(foldBoolBinary<XorBoolOp>(boolAttr(false), boolAttr(false)), false);
}

//===------------------------------------------------------------------===//
// bool.not
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, NotTrue) { expectBool(foldBoolUnary<NotBoolOp>(boolAttr(true)), false); }

TEST_F(BoolFoldTest, NotFalse) { expectBool(foldBoolUnary<NotBoolOp>(boolAttr(false)), true); }

//===------------------------------------------------------------------===//
// bool.cmp — equality / inequality
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, CmpEqSame) {
  expectBool(foldCmp(FeltCmpPredicate::EQ, babyBearConst(5), babyBearConst(5)), true);
}

TEST_F(BoolFoldTest, CmpEqDiff) {
  expectBool(foldCmp(FeltCmpPredicate::EQ, babyBearConst(3), babyBearConst(7)), false);
}

TEST_F(BoolFoldTest, CmpNeDiff) {
  expectBool(foldCmp(FeltCmpPredicate::NE, babyBearConst(3), babyBearConst(7)), true);
}

TEST_F(BoolFoldTest, CmpNeSame) {
  expectBool(foldCmp(FeltCmpPredicate::NE, babyBearConst(5), babyBearConst(5)), false);
}

//===------------------------------------------------------------------===//
// bool.cmp — ordering (unsigned integer comparison of field elements)
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, CmpLtTrue) {
  expectBool(foldCmp(FeltCmpPredicate::LT, babyBearConst(3), babyBearConst(7)), true);
}

TEST_F(BoolFoldTest, CmpLtFalse) {
  expectBool(foldCmp(FeltCmpPredicate::LT, babyBearConst(7), babyBearConst(3)), false);
}

TEST_F(BoolFoldTest, CmpLtEqual) {
  expectBool(foldCmp(FeltCmpPredicate::LT, babyBearConst(5), babyBearConst(5)), false);
}

TEST_F(BoolFoldTest, CmpLeEqual) {
  expectBool(foldCmp(FeltCmpPredicate::LE, babyBearConst(5), babyBearConst(5)), true);
}

TEST_F(BoolFoldTest, CmpLeLess) {
  expectBool(foldCmp(FeltCmpPredicate::LE, babyBearConst(3), babyBearConst(7)), true);
}

TEST_F(BoolFoldTest, CmpLeGreater) {
  expectBool(foldCmp(FeltCmpPredicate::LE, babyBearConst(7), babyBearConst(3)), false);
}

TEST_F(BoolFoldTest, CmpGtTrue) {
  expectBool(foldCmp(FeltCmpPredicate::GT, babyBearConst(7), babyBearConst(3)), true);
}

TEST_F(BoolFoldTest, CmpGtFalse) {
  expectBool(foldCmp(FeltCmpPredicate::GT, babyBearConst(3), babyBearConst(7)), false);
}

TEST_F(BoolFoldTest, CmpGeEqual) {
  expectBool(foldCmp(FeltCmpPredicate::GE, babyBearConst(5), babyBearConst(5)), true);
}

TEST_F(BoolFoldTest, CmpGeGreater) {
  expectBool(foldCmp(FeltCmpPredicate::GE, babyBearConst(7), babyBearConst(3)), true);
}

TEST_F(BoolFoldTest, CmpGeFalse) {
  expectBool(foldCmp(FeltCmpPredicate::GE, babyBearConst(3), babyBearConst(7)), false);
}

//===------------------------------------------------------------------===//
// No-fold cases
//===------------------------------------------------------------------===//

TEST_F(BoolFoldTest, AndNoFoldNonConst) {
  // Pass null attributes to simulate non-constant operands
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPoint(&block, block.end());
  auto i1Ty = builder.getI1Type();
  // Create block arguments (non-constant)
  auto arg0 = block.addArgument(i1Ty, loc);
  auto arg1 = block.addArgument(i1Ty, loc);
  auto op = builder.create<AndBoolOp>(loc, arg0, arg1);

  SmallVector<Attribute, 2> operands = {Attribute(), Attribute()};
  SmallVector<OpFoldResult, 1> results;
  // Fold should either fail or produce an empty result when operands are unknown
  bool folded = succeeded(op->fold(operands, results)) && !results.empty() &&
                dyn_cast_if_present<Attribute>(results[0]);
  EXPECT_FALSE(folded) << "expected fold to be skipped for non-constant operands";
}

TEST_F(BoolFoldTest, CmpNoFoldNonConst) {
  // Pass null felt attributes to simulate non-constant felt operands
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPoint(&block, block.end());
  auto feltTy = FeltType::get(&ctx, StringAttr::get(&ctx, BB_FIELD));
  auto arg0 = block.addArgument(feltTy, loc);
  auto arg1 = block.addArgument(feltTy, loc);
  auto predAttr = FeltCmpPredicateAttr::get(&ctx, FeltCmpPredicate::EQ);
  auto cmpOp = builder.create<CmpOp>(loc, predAttr, arg0, arg1);

  SmallVector<Attribute, 2> operands = {Attribute(), Attribute()};
  SmallVector<OpFoldResult, 1> results;
  bool folded = succeeded(cmpOp->fold(operands, results)) && !results.empty() &&
                dyn_cast_if_present<Attribute>(results[0]);
  EXPECT_FALSE(folded) << "expected fold to be skipped for non-constant operands";
}
