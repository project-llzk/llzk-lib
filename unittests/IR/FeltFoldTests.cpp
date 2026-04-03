//===-- FeltFoldTests.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

#include <gtest/gtest.h>

#include "../LLZKTestBase.h"

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;

//===------------------------------------------------------------------===//
// Test infrastructure
//===------------------------------------------------------------------===//

class FeltFoldTest : public LLZKTest {
protected:
  // babybear: p = 2013265921, bitWidth = 32, half = (p+1)/2 = 1006632961
  static constexpr uint64_t BB_PRIME = 2013265921ULL;
  static constexpr unsigned BB_BITS = 32;
  static constexpr StringLiteral BB_FIELD = "babybear";

  FeltConstAttr bbConst(uint64_t val) {
    return FeltConstAttr::get(&ctx, APInt(BB_BITS, val), BB_FIELD);
  }

  FeltConstAttr noFieldConst(uint64_t val) { return FeltConstAttr::get(&ctx, APInt(BB_BITS, val)); }

  /// Insert two `felt.const` ops into a detached block, build a binary op on
  /// them, call fold with the constant attributes, and return the folded
  /// FeltConstAttr (or null if the fold did not produce a constant).
  template <typename OpTy> FeltConstAttr foldBinary(FeltConstAttr lhsAttr, FeltConstAttr rhsAttr) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    auto lhsOp = builder.create<FeltConstantOp>(loc, lhsAttr);
    auto rhsOp = builder.create<FeltConstantOp>(loc, rhsAttr);
    auto op = builder.create<OpTy>(loc, lhsOp.getResult(), rhsOp.getResult());

    SmallVector<Attribute, 2> operands = {lhsAttr, rhsAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(op->fold(operands, results)) || results.empty()) {
      return {};
    }
    return llvm::dyn_cast_or_null<FeltConstAttr>(llvm::dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Same as foldBinary but for unary ops.
  template <typename OpTy> FeltConstAttr foldUnary(FeltConstAttr operandAttr) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    auto operandOp = builder.create<FeltConstantOp>(loc, operandAttr);
    auto op = builder.create<OpTy>(loc, operandOp.getResult());

    SmallVector<Attribute, 1> operands = {operandAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(op->fold(operands, results)) || results.empty()) {
      return {};
    }
    return llvm::dyn_cast_or_null<FeltConstAttr>(llvm::dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Assert that fold produced the expected unsigned integer value in babybear.
  void expectValue(FeltConstAttr result, uint64_t expected) {
    ASSERT_TRUE(result) << "expected fold to succeed";
    EXPECT_EQ(result.getValue().getZExtValue(), expected);
    EXPECT_EQ(result.getFieldName(), StringAttr::get(&ctx, BB_FIELD));
  }

  /// Assert that no fold occurred (fold returned null/empty).
  void expectNoFold(FeltConstAttr result) { EXPECT_FALSE(result) << "expected fold to be skipped"; }
};

//===------------------------------------------------------------------===//
// felt.add
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, AddSimple) { expectValue(foldBinary<AddFeltOp>(bbConst(5), bbConst(7)), 12); }

TEST_F(FeltFoldTest, AddWrapAround) {
  // (p-1) + 2 = p+1 ≡ 1 (mod p)
  expectValue(foldBinary<AddFeltOp>(bbConst(BB_PRIME - 1), bbConst(2)), 1);
}

TEST_F(FeltFoldTest, AddNoFoldUnspecified) {
  expectNoFold(foldBinary<AddFeltOp>(noFieldConst(5), noFieldConst(7)));
}

//===------------------------------------------------------------------===//
// felt.sub
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, SubSimple) { expectValue(foldBinary<SubFeltOp>(bbConst(10), bbConst(7)), 3); }

TEST_F(FeltFoldTest, SubWrapAround) {
  // 3 - 10 = -7 ≡ p-7 = 2013265914 (mod p)
  expectValue(foldBinary<SubFeltOp>(bbConst(3), bbConst(10)), BB_PRIME - 7);
}

TEST_F(FeltFoldTest, SubNoFoldUnspecified) {
  expectNoFold(foldBinary<SubFeltOp>(noFieldConst(10), noFieldConst(3)));
}

//===------------------------------------------------------------------===//
// felt.mul
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, MulSimple) { expectValue(foldBinary<MulFeltOp>(bbConst(6), bbConst(7)), 42); }

TEST_F(FeltFoldTest, MulWrapAround) {
  // half(p) * 2 = (p+1)/2 * 2 = p+1 ≡ 1 (mod p)
  expectValue(foldBinary<MulFeltOp>(bbConst(1006632961), bbConst(2)), 1);
}

//===------------------------------------------------------------------===//
// felt.pow
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, PowSimple) {
  expectValue(foldBinary<PowFeltOp>(bbConst(2), bbConst(10)), 1024);
}

TEST_F(FeltFoldTest, PowByZero) {
  // Any value to the power 0 = 1
  expectValue(foldBinary<PowFeltOp>(bbConst(7), bbConst(0)), 1);
}

//===------------------------------------------------------------------===//
// felt.div
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, DivSimple) {
  // 6 / 2 = 6 * inv(2) mod p = 6 * 1006632961 mod p = 3
  expectValue(foldBinary<DivFeltOp>(bbConst(6), bbConst(2)), 3);
}

TEST_F(FeltFoldTest, DivByZeroNoFold) {
  expectNoFold(foldBinary<DivFeltOp>(bbConst(5), bbConst(0)));
}

//===------------------------------------------------------------------===//
// felt.uintdiv
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, UintDivSimple) {
  // 7 / 2 = 3 (truncating unsigned)
  expectValue(foldBinary<UnsignedIntDivFeltOp>(bbConst(7), bbConst(2)), 3);
}

TEST_F(FeltFoldTest, UintDivExact) {
  expectValue(foldBinary<UnsignedIntDivFeltOp>(bbConst(12), bbConst(4)), 3);
}

TEST_F(FeltFoldTest, UintDivByZeroNoFold) {
  expectNoFold(foldBinary<UnsignedIntDivFeltOp>(bbConst(5), bbConst(0)));
}

//===------------------------------------------------------------------===//
// felt.sintdiv
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, SintDivPositive) {
  // Both positive: 7 / 2 = 3
  expectValue(foldBinary<SignedIntDivFeltOp>(bbConst(7), bbConst(2)), 3);
}

TEST_F(FeltFoldTest, SintDivNegativeDividend) {
  // (p-7) is signed -7; -7 / 2 = -3 (truncate); reduce(-3) = p-3 = 2013265918
  expectValue(foldBinary<SignedIntDivFeltOp>(bbConst(BB_PRIME - 7), bbConst(2)), BB_PRIME - 3);
}

TEST_F(FeltFoldTest, SintDivByZeroNoFold) {
  expectNoFold(foldBinary<SignedIntDivFeltOp>(bbConst(5), bbConst(0)));
}

//===------------------------------------------------------------------===//
// felt.umod
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, UmodSimple) {
  expectValue(foldBinary<UnsignedModFeltOp>(bbConst(7), bbConst(2)), 1);
}

TEST_F(FeltFoldTest, UmodExact) {
  expectValue(foldBinary<UnsignedModFeltOp>(bbConst(12), bbConst(4)), 0);
}

TEST_F(FeltFoldTest, UmodByZeroNoFold) {
  expectNoFold(foldBinary<UnsignedModFeltOp>(bbConst(5), bbConst(0)));
}

//===------------------------------------------------------------------===//
// felt.smod
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, SmodPositive) {
  expectValue(foldBinary<SignedModFeltOp>(bbConst(7), bbConst(2)), 1);
}

TEST_F(FeltFoldTest, SmodNegativeDividend) {
  // (p-7) is signed -7; -7 % 2 = -1 (truncating); reduce(-1) = p-1 = 2013265920
  expectValue(foldBinary<SignedModFeltOp>(bbConst(BB_PRIME - 7), bbConst(2)), BB_PRIME - 1);
}

TEST_F(FeltFoldTest, SmodByZeroNoFold) {
  expectNoFold(foldBinary<SignedModFeltOp>(bbConst(5), bbConst(0)));
}

//===------------------------------------------------------------------===//
// felt.neg
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, NegSimple) {
  // neg 5 = p - 5 = 2013265916
  expectValue(foldUnary<NegFeltOp>(bbConst(5)), BB_PRIME - 5);
}

TEST_F(FeltFoldTest, NegZero) { expectValue(foldUnary<NegFeltOp>(bbConst(0)), 0); }

TEST_F(FeltFoldTest, NegNoFoldUnspecified) { expectNoFold(foldUnary<NegFeltOp>(noFieldConst(5))); }

//===------------------------------------------------------------------===//
// felt.inv
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, InvOne) { expectValue(foldUnary<InvFeltOp>(bbConst(1)), 1); }

TEST_F(FeltFoldTest, InvTwo) {
  // inv(2) = (p+1)/2 = 1006632961 in babybear
  expectValue(foldUnary<InvFeltOp>(bbConst(2)), 1006632961);
}

TEST_F(FeltFoldTest, InvZeroNoFold) { expectNoFold(foldUnary<InvFeltOp>(bbConst(0))); }

TEST_F(FeltFoldTest, InvNoFoldUnspecified) { expectNoFold(foldUnary<InvFeltOp>(noFieldConst(2))); }

//===------------------------------------------------------------------===//
// felt.bit_and / felt.bit_or / felt.bit_xor
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, BitAnd) {
  // 0b110 & 0b011 = 0b010 = 2
  expectValue(foldBinary<AndFeltOp>(bbConst(6), bbConst(3)), 2);
}

TEST_F(FeltFoldTest, BitOr) {
  // 0b110 | 0b011 = 0b111 = 7
  expectValue(foldBinary<OrFeltOp>(bbConst(6), bbConst(3)), 7);
}

TEST_F(FeltFoldTest, BitXor) {
  // 0b110 ^ 0b011 = 0b101 = 5
  expectValue(foldBinary<XorFeltOp>(bbConst(6), bbConst(3)), 5);
}

//===------------------------------------------------------------------===//
// felt.shl / felt.shr
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, Shl) {
  // 1 << 4 = 16
  expectValue(foldBinary<ShlFeltOp>(bbConst(1), bbConst(4)), 16);
}

TEST_F(FeltFoldTest, ShlWrap) {
  // Shift that wraps: 1 << 31 = 2147483648; reduce mod p
  // 2147483648 mod 2013265921 = 134217727
  expectValue(foldBinary<ShlFeltOp>(bbConst(1), bbConst(31)), 2147483648ULL % BB_PRIME);
}

TEST_F(FeltFoldTest, Shr) {
  // 16 >> 2 = 4
  expectValue(foldBinary<ShrFeltOp>(bbConst(16), bbConst(2)), 4);
}

//===------------------------------------------------------------------===//
// felt.bit_not
//===------------------------------------------------------------------===//

TEST_F(FeltFoldTest, BitNotZero) {
  // ~0 at bitWidth=31: reduce(2^31 - 1) = 2147483647 - p = 134217726
  expectValue(foldUnary<NotFeltOp>(bbConst(0)), 134217726);
}

TEST_F(FeltFoldTest, BitNotNoFoldUnspecified) {
  expectNoFold(foldUnary<NotFeltOp>(noFieldConst(0)));
}
