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

class BabyBearFoldTest : public LLZKTest {
protected:
  /// bitwidth large enough to hold babybear prime (which is 31 bits) with optional sign bit.
  static constexpr unsigned BITWIDTH = 32;
  /// babybear: p = 2013265921, half = (p+1)/2 = 1006632961
  static constexpr uint64_t BB_PRIME = 2013265921ULL;
  static constexpr StringLiteral BB_FIELD = "babybear";

  FeltConstAttr babyBearConst(uint64_t val) {
    return FeltConstAttr::get(&ctx, APInt(BITWIDTH, val), BB_FIELD);
  }

  FeltConstAttr unspecifiedConst(uint64_t val) {
    return FeltConstAttr::get(&ctx, APInt(BITWIDTH, val));
  }

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

TEST_F(BabyBearFoldTest, AddSimple) {
  expectValue(foldBinary<AddFeltOp>(babyBearConst(5), babyBearConst(7)), 12);
}

TEST_F(BabyBearFoldTest, AddWrapAround) {
  // (p-1) + 2 = p+1 ≡ 1 (mod p)
  expectValue(foldBinary<AddFeltOp>(babyBearConst(BB_PRIME - 1), babyBearConst(2)), 1);
}

TEST_F(BabyBearFoldTest, AddNoFoldUnspecified) {
  expectNoFold(foldBinary<AddFeltOp>(unspecifiedConst(5), unspecifiedConst(7)));
}

//===------------------------------------------------------------------===//
// felt.sub
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, SubSimple) {
  expectValue(foldBinary<SubFeltOp>(babyBearConst(10), babyBearConst(7)), 3);
}

TEST_F(BabyBearFoldTest, SubWrapAround) {
  // 3 - 10 = -7 ≡ p-7 = 2013265914 (mod p)
  expectValue(foldBinary<SubFeltOp>(babyBearConst(3), babyBearConst(10)), BB_PRIME - 7);
}

TEST_F(BabyBearFoldTest, SubNoFoldUnspecified) {
  expectNoFold(foldBinary<SubFeltOp>(unspecifiedConst(10), unspecifiedConst(3)));
}

//===------------------------------------------------------------------===//
// felt.mul
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, MulSimple) {
  expectValue(foldBinary<MulFeltOp>(babyBearConst(6), babyBearConst(7)), 42);
}

TEST_F(BabyBearFoldTest, MulWrapAround) {
  // half(p) * 2 = (p+1)/2 * 2 = p+1 ≡ 1 (mod p)
  expectValue(foldBinary<MulFeltOp>(babyBearConst(1006632961), babyBearConst(2)), 1);
}

//===------------------------------------------------------------------===//
// felt.pow
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, PowSimple) {
  expectValue(foldBinary<PowFeltOp>(babyBearConst(2), babyBearConst(10)), 1024);
}

TEST_F(BabyBearFoldTest, PowByZero) {
  // Any value to the power 0 = 1
  expectValue(foldBinary<PowFeltOp>(babyBearConst(7), babyBearConst(0)), 1);
}

//===------------------------------------------------------------------===//
// felt.div
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, DivSimple) {
  // 6 / 2 = 6 * inv(2) mod p = 6 * 1006632961 mod p = 3
  expectValue(foldBinary<DivFeltOp>(babyBearConst(6), babyBearConst(2)), 3);
}

TEST_F(BabyBearFoldTest, DivByZeroNoFold) {
  expectNoFold(foldBinary<DivFeltOp>(babyBearConst(5), babyBearConst(0)));
}

//===------------------------------------------------------------------===//
// felt.uintdiv
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, UintDivSimple) {
  // 7 / 2 = 3 (truncating unsigned)
  expectValue(foldBinary<UnsignedIntDivFeltOp>(babyBearConst(7), babyBearConst(2)), 3);
}

TEST_F(BabyBearFoldTest, UintDivExact) {
  expectValue(foldBinary<UnsignedIntDivFeltOp>(babyBearConst(12), babyBearConst(4)), 3);
}

TEST_F(BabyBearFoldTest, UintDivByZeroNoFold) {
  expectNoFold(foldBinary<UnsignedIntDivFeltOp>(babyBearConst(5), babyBearConst(0)));
}

//===------------------------------------------------------------------===//
// felt.sintdiv
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, SintDivPositive) {
  // Both positive: 7 / 2 = 3
  expectValue(foldBinary<SignedIntDivFeltOp>(babyBearConst(7), babyBearConst(2)), 3);
}

TEST_F(BabyBearFoldTest, SintDivNegativeDividend) {
  // (p-7) is signed -7; -7 / 2 = -3 (truncate); reduce(-3) = p-3 = 2013265918
  expectValue(
      foldBinary<SignedIntDivFeltOp>(babyBearConst(BB_PRIME - 7), babyBearConst(2)), BB_PRIME - 3
  );
}

TEST_F(BabyBearFoldTest, SintDivByPrimeNoFold) {
  expectNoFold(foldBinary<SignedIntDivFeltOp>(babyBearConst(89735), babyBearConst(BB_PRIME)));
}

TEST_F(BabyBearFoldTest, SintDivByZeroNoFold) {
  expectNoFold(foldBinary<SignedIntDivFeltOp>(babyBearConst(5), babyBearConst(0)));
}

//===------------------------------------------------------------------===//
// felt.umod
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, UmodSimple) {
  expectValue(foldBinary<UnsignedModFeltOp>(babyBearConst(7), babyBearConst(2)), 1);
}

TEST_F(BabyBearFoldTest, UmodExact) {
  expectValue(foldBinary<UnsignedModFeltOp>(babyBearConst(12), babyBearConst(4)), 0);
}

TEST_F(BabyBearFoldTest, UmodByZeroNoFold) {
  expectNoFold(foldBinary<UnsignedModFeltOp>(babyBearConst(5), babyBearConst(0)));
}

//===------------------------------------------------------------------===//
// felt.smod
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, SmodPositive) {
  expectValue(foldBinary<SignedModFeltOp>(babyBearConst(7), babyBearConst(2)), 1);
}

TEST_F(BabyBearFoldTest, SmodNegativeDividend) {
  // (p-7) is signed -7; -7 % 2 = -1 (truncating); reduce(-1) = p-1 = 2013265920
  expectValue(
      foldBinary<SignedModFeltOp>(babyBearConst(BB_PRIME - 7), babyBearConst(2)), BB_PRIME - 1
  );
}

TEST_F(BabyBearFoldTest, SmodByPrimeNoFold) {
  expectNoFold(foldBinary<SignedModFeltOp>(babyBearConst(1865), babyBearConst(BB_PRIME)));
}

TEST_F(BabyBearFoldTest, SmodByZeroNoFold) {
  expectNoFold(foldBinary<SignedModFeltOp>(babyBearConst(5), babyBearConst(0)));
}

//===------------------------------------------------------------------===//
// felt.neg
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, NegSimple) {
  // neg 5 = p - 5 = 2013265916
  expectValue(foldUnary<NegFeltOp>(babyBearConst(5)), BB_PRIME - 5);
}

TEST_F(BabyBearFoldTest, NegZero) { expectValue(foldUnary<NegFeltOp>(babyBearConst(0)), 0); }

TEST_F(BabyBearFoldTest, NegNoFoldUnspecified) {
  expectNoFold(foldUnary<NegFeltOp>(unspecifiedConst(5)));
}

//===------------------------------------------------------------------===//
// felt.inv
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, InvOne) { expectValue(foldUnary<InvFeltOp>(babyBearConst(1)), 1); }

TEST_F(BabyBearFoldTest, InvTwo) {
  // inv(2) = (p+1)/2 = 1006632961 in babybear
  expectValue(foldUnary<InvFeltOp>(babyBearConst(2)), 1006632961);
}

TEST_F(BabyBearFoldTest, InvZeroNoFold) { expectNoFold(foldUnary<InvFeltOp>(babyBearConst(0))); }

TEST_F(BabyBearFoldTest, InvNoFoldUnspecified) {
  expectNoFold(foldUnary<InvFeltOp>(unspecifiedConst(2)));
}

//===------------------------------------------------------------------===//
// felt.bit_and / felt.bit_or / felt.bit_xor
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, BitAnd) {
  // 0b110 & 0b011 = 0b010 = 2
  expectValue(foldBinary<AndFeltOp>(babyBearConst(6), babyBearConst(3)), 2);
}

TEST_F(BabyBearFoldTest, BitOr) {
  // 0b110 | 0b011 = 0b111 = 7
  expectValue(foldBinary<OrFeltOp>(babyBearConst(6), babyBearConst(3)), 7);
}

TEST_F(BabyBearFoldTest, BitXor) {
  // 0b110 ^ 0b011 = 0b101 = 5
  expectValue(foldBinary<XorFeltOp>(babyBearConst(6), babyBearConst(3)), 5);
}

//===------------------------------------------------------------------===//
// felt.shl / felt.shr
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, Shl) {
  // 1 << 4 = 16
  expectValue(foldBinary<ShlFeltOp>(babyBearConst(1), babyBearConst(4)), 16);
}

TEST_F(BabyBearFoldTest, Shl31) {
  // Shift that wraps: 1 << 31 = 2147483648; reduce mod p
  // 2147483648 mod 2013265921 = 134217727
  expectValue(foldBinary<ShlFeltOp>(babyBearConst(1), babyBearConst(31)), 134217727ULL);
}

TEST_F(BabyBearFoldTest, Shl32) {
  // Shift that wraps: 1 << 32 = 4294967296; reduce mod p
  // 4294967296 mod 2013265921 = 268435454
  expectValue(foldBinary<ShlFeltOp>(babyBearConst(1), babyBearConst(32)), 268435454ULL);
}

TEST_F(BabyBearFoldTest, Shl33) {
  // Shift that wraps: 1 << 33 = 8589934592; reduce mod p
  // 8589934592 mod 2013265921 = 536870908
  expectValue(foldBinary<ShlFeltOp>(babyBearConst(1), babyBearConst(33)), 536870908ULL);
}

TEST_F(BabyBearFoldTest, Shr) {
  // 16 >> 2 = 4
  expectValue(foldBinary<ShrFeltOp>(babyBearConst(16), babyBearConst(2)), 4);
}

TEST_F(BabyBearFoldTest, Shr30) {
  // BB_PRIME >> 30 = 1 (only highest bit remains, all others shifted out)
  expectValue(foldBinary<ShrFeltOp>(babyBearConst(BB_PRIME), babyBearConst(30)), 1);
}

TEST_F(BabyBearFoldTest, Shr31) {
  // BB_PRIME >> 31 = 0 (shifts all bits out since BB_PRIME is 31 bits)
  expectValue(foldBinary<ShrFeltOp>(babyBearConst(BB_PRIME), babyBearConst(31)), 0);
}

//===------------------------------------------------------------------===//
// felt.bit_not
//===------------------------------------------------------------------===//

TEST_F(BabyBearFoldTest, BitNotZero) {
  // ~0 at bitWidth=31: reduce(2^31 - 1) = 2147483647 - p = 134217726
  expectValue(foldUnary<NotFeltOp>(babyBearConst(0)), 134217726);
}

TEST_F(BabyBearFoldTest, BitNotNoFoldUnspecified) {
  expectNoFold(foldUnary<NotFeltOp>(unspecifiedConst(0)));
}
