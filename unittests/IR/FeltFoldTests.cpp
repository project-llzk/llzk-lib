//===-- FeltFoldTests.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

#include <array>
#include <gtest/gtest.h>
#include <limits>

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
    return FeltConstAttr::get(&ctx, toDynamicAPInt(val), BB_FIELD);
  }

  FeltConstAttr unspecifiedConst(uint64_t val) {
    return FeltConstAttr::get(&ctx, toDynamicAPInt(val));
  }

  /// Insert two `felt.const` ops into a detached block, build a binary op on
  /// them, call fold with the constant attributes, and return the folded
  /// FeltConstAttr (or null if the fold did not produce a constant).
  template <typename OpTy> FeltConstAttr foldBinary(FeltConstAttr lhsAttr, FeltConstAttr rhsAttr) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    auto lhsOp = FeltConstantOp::create(builder, loc, lhsAttr);
    auto rhsOp = FeltConstantOp::create(builder, loc, rhsAttr);
    auto op = OpTy::create(builder, loc, lhsOp.getResult(), rhsOp.getResult());

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

    auto operandOp = FeltConstantOp::create(builder, loc, operandAttr);
    auto op = OpTy::create(builder, loc, operandOp.getResult());

    SmallVector<Attribute, 1> operands = {operandAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(op->fold(operands, results)) || results.empty()) {
      return {};
    }
    return llvm::dyn_cast_or_null<FeltConstAttr>(llvm::dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Fold a cast from a block argument using only the supplied constant attribute.
  IntegerAttr foldToIndex(
      FeltConstAttr operandAttr,
      llzk::cast::OverflowSemantics overflow = llzk::cast::OverflowSemantics::ASSERT
  ) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPointToEnd(&block);
    auto type = operandAttr ? operandAttr.getType() : FeltType::get(&ctx, BB_FIELD);
    auto operand = block.addArgument(type, loc);
    auto op = llzk::cast::FeltToIndexOp::create(builder, loc, overflow, operand);
    SmallVector<Attribute, 1> operands = {operandAttr};
    SmallVector<OpFoldResult, 1> results;
    if (failed(op->fold(operands, results)) || results.empty()) {
      return {};
    }
    return llvm::dyn_cast_or_null<IntegerAttr>(llvm::dyn_cast_if_present<Attribute>(results[0]));
  }

  /// Fold an integer-to-felt cast using an adaptor constant and a block argument.
  FeltConstAttr foldToFelt(
      IntegerAttr operandAttr, FeltType resultType,
      llzk::cast::OverflowSemantics overflow = llzk::cast::OverflowSemantics::ASSERT
  ) {
    Block block;
    OpBuilder builder(&ctx);
    builder.setInsertionPointToEnd(&block);
    auto type = operandAttr ? operandAttr.getType() : builder.getIndexType();
    auto operand = block.addArgument(type, loc);
    auto op = llzk::cast::IntToFeltOp::create(builder, loc, resultType, overflow, operand);
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
    EXPECT_EQ(result.getRawValue(), toDynamicAPInt(expected));
    EXPECT_EQ(result.getFieldName(), StringAttr::get(&ctx, BB_FIELD));
  }

  /// Assert that no fold occurred (fold returned null/empty).
  void expectNoFold(FeltConstAttr result) { EXPECT_FALSE(result) << "expected fold to be skipped"; }
};

TEST_F(BabyBearFoldTest, ReducedConstantBoundaries) {
  for (uint64_t value :
       std::array<uint64_t, 6> {0, 1, BB_PRIME - 1, BB_PRIME, BB_PRIME + 1, 2 * BB_PRIME}) {
    auto attr = babyBearConst(value);
    ASSERT_TRUE(attr.getReducedValue());
    EXPECT_EQ(*attr.getReducedValue(), value % BB_PRIME);
    // Exercise the fallback accessor's reduced branch for a specified field.
    EXPECT_EQ(attr.getReducedValueOrRaw(), value % BB_PRIME);
    EXPECT_EQ(attr.getRawValue(), toDynamicAPInt(value));
  }
}

TEST_F(BabyBearFoldTest, ReducedConstantUsesSignedValue) {
  auto attr = FeltConstAttr::get(&ctx, llvm::DynamicAPInt(-1), BB_FIELD);
  ASSERT_TRUE(attr.getReducedValue());
  EXPECT_EQ(*attr.getReducedValue(), toDynamicAPInt(BB_PRIME - 1));
  // Verify the fallback accessor also reduces signed literals in a known field.
  EXPECT_EQ(attr.getReducedValueOrRaw(), toDynamicAPInt(BB_PRIME - 1));
  EXPECT_EQ(attr.getRawValue(), -1);
}

TEST_F(BabyBearFoldTest, ReducedConstantHandlesWideValues) {
  llvm::DynamicAPInt prime = toDynamicAPInt(BB_PRIME);
  llvm::DynamicAPInt raw = prime * toDynamicAPInt(APInt::getOneBitSet(256, 128)) + 17;
  auto attr = FeltConstAttr::get(&ctx, raw, BB_FIELD);
  ASSERT_TRUE(attr.getReducedValue());
  EXPECT_EQ(*attr.getReducedValue(), 17U);
  EXPECT_EQ(attr.getRawValue(), raw);
}

TEST_F(BabyBearFoldTest, ReducedConstantWithoutFieldIsAbsent) {
  llvm::DynamicAPInt raw = toDynamicAPInt(APInt::getAllOnes(256));
  auto attr = FeltConstAttr::get(&ctx, raw);
  EXPECT_FALSE(attr.getReducedValue());
  // Exercise the raw fallback and ensure a wide positive literal keeps its sign.
  EXPECT_EQ(attr.getReducedValueOrRaw(), raw);
  EXPECT_GT(attr.getReducedValueOrRaw(), 0);
  EXPECT_EQ(attr.getRawValue(), raw);
}

TEST_F(BabyBearFoldTest, NegativeFieldlessConstantPreservesRawValue) {
  auto raw = toDynamicAPInt("-18446744073709551617");
  auto attr = FeltConstAttr::get(&ctx, raw);
  EXPECT_FALSE(attr.getReducedValue());
  // Exercise the raw fallback for a negative literal with no field to reduce it.
  EXPECT_EQ(attr.getReducedValueOrRaw(), raw);
  EXPECT_EQ(attr.getRawValue(), raw);
}

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

TEST_F(BabyBearFoldTest, Shl100) {
  expectValue(foldBinary<ShlFeltOp>(babyBearConst(1), babyBearConst(100)), 1060618158ULL);
}

TEST_F(BabyBearFoldTest, Shr) {
  // 16 >> 2 = 4
  expectValue(foldBinary<ShrFeltOp>(babyBearConst(16), babyBearConst(2)), 4);
}

TEST_F(BabyBearFoldTest, Shr30) {
  // (BB_PRIME - 1) >> 30 = 1 (only the highest bit remains).
  expectValue(foldBinary<ShrFeltOp>(babyBearConst(BB_PRIME - 1), babyBearConst(30)), 1);
}

TEST_F(BabyBearFoldTest, Shr31) {
  // (BB_PRIME - 1) >> 31 = 0 (all 31 bits are shifted out).
  expectValue(foldBinary<ShrFeltOp>(babyBearConst(BB_PRIME - 1), babyBearConst(31)), 0);
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

TEST_F(BabyBearFoldTest, FoldsReduceUncanonicalizedOperands) {
  auto one = babyBearConst(BB_PRIME + 1);
  auto two = babyBearConst(2);
  auto zero = babyBearConst(BB_PRIME);
  expectValue(foldBinary<ShrFeltOp>(zero, babyBearConst(30)), 0);
  expectValue(foldBinary<UnsignedIntDivFeltOp>(one, two), 0);
  expectValue(foldBinary<UnsignedModFeltOp>(one, two), 1);
  expectValue(foldBinary<SignedIntDivFeltOp>(one, two), 0);
  expectValue(foldBinary<SignedModFeltOp>(one, two), 1);
  expectValue(foldBinary<AndFeltOp>(one, two), 0);
  expectValue(foldBinary<OrFeltOp>(one, two), 3);
  expectValue(foldBinary<PowFeltOp>(two, one), 2);
  expectNoFold(foldBinary<DivFeltOp>(two, zero));
  expectNoFold(foldBinary<UnsignedIntDivFeltOp>(two, zero));
  expectNoFold(foldBinary<UnsignedModFeltOp>(two, zero));
}

// Verify that casting a felt constant uses its mathematical value without requiring the
// constant to be canonicalized first.
TEST_F(BabyBearFoldTest, CastToIndexReducesWithoutCanonicalizingConstant) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto constant = FeltConstantOp::create(builder, loc, babyBearConst(BB_PRIME + 7));
  ASSERT_TRUE(constant.getValue().getReducedValue());
  EXPECT_EQ(*constant.getValue().getReducedValue(), 7U);
  Value folded = builder.createOrFold<llzk::cast::FeltToIndexOp>(loc, constant.getResult());
  auto index = folded.getDefiningOp<arith::ConstantIndexOp>();
  ASSERT_TRUE(index);
  EXPECT_EQ(index.value(), 7);
  EXPECT_EQ(constant.getValue().getRawValue(), toDynamicAPInt(BB_PRIME + 7));
}

TEST_F(BabyBearFoldTest, CastToIndexFoldsAdaptorConstant) {
  auto result = foldToIndex(babyBearConst(BB_PRIME + 7));
  ASSERT_TRUE(result);
  EXPECT_TRUE(result.getType().isIndex());
  EXPECT_EQ(result.getInt(), 7);
}

TEST_F(BabyBearFoldTest, CastToIndexDefersUnknownConstants) {
  for (auto mode :
       {llzk::cast::OverflowSemantics::ASSERT, llzk::cast::OverflowSemantics::SATURATE,
        llzk::cast::OverflowSemantics::WRAP, llzk::cast::OverflowSemantics::TRUNCATE}) {
    EXPECT_FALSE(foldToIndex({}, mode));
    EXPECT_FALSE(foldToIndex(unspecifiedConst(0), mode));
    EXPECT_FALSE(foldToIndex(unspecifiedConst(BB_PRIME + 7), mode));
  }
}

TEST_F(BabyBearFoldTest, CastToIndexFoldPreservesOverflowSemantics) {
  const int64_t maxIndex = std::numeric_limits<int64_t>::max();
  auto maxValue = FeltConstAttr::get(&ctx, llvm::DynamicAPInt(maxIndex), "bn254");
  auto tooLarge = FeltConstAttr::get(&ctx, llvm::DynamicAPInt(maxIndex) + 13, "bn254");
  auto result = foldToIndex(maxValue);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.getInt(), maxIndex);
  EXPECT_FALSE(foldToIndex(tooLarge));

  result = foldToIndex(tooLarge, llzk::cast::OverflowSemantics::SATURATE);
  ASSERT_TRUE(result);
  EXPECT_EQ(result.getInt(), maxIndex);
  for (auto mode : {llzk::cast::OverflowSemantics::WRAP, llzk::cast::OverflowSemantics::TRUNCATE}) {
    result = foldToIndex(tooLarge, mode);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.getInt(), 12);
  }
}

TEST_F(BabyBearFoldTest, CastToFeltFoldsAdaptorConstants) {
  OpBuilder builder(&ctx);
  auto type = FeltType::get(&ctx, BB_FIELD);
  expectValue(foldToFelt(builder.getIndexAttr(7), type), 7);
  for (auto mode :
       {llzk::cast::OverflowSemantics::ASSERT, llzk::cast::OverflowSemantics::SATURATE,
        llzk::cast::OverflowSemantics::WRAP, llzk::cast::OverflowSemantics::TRUNCATE}) {
    expectValue(foldToFelt(builder.getBoolAttr(false), type, mode), 0);
    expectValue(foldToFelt(builder.getBoolAttr(true), type, mode), 1);
  }
}

TEST_F(BabyBearFoldTest, CastToFeltDefersUnknownConstantsAndFields) {
  OpBuilder builder(&ctx);
  for (auto mode :
       {llzk::cast::OverflowSemantics::ASSERT, llzk::cast::OverflowSemantics::SATURATE,
        llzk::cast::OverflowSemantics::WRAP, llzk::cast::OverflowSemantics::TRUNCATE}) {
    expectNoFold(foldToFelt({}, FeltType::get(&ctx, BB_FIELD), mode));
    expectNoFold(foldToFelt(builder.getIndexAttr(0), FeltType::get(&ctx), mode));
    expectNoFold(foldToFelt(builder.getBoolAttr(true), FeltType::get(&ctx), mode));
  }
}

TEST_F(BabyBearFoldTest, CastToFeltFoldPreservesOverflowSemantics) {
  OpBuilder builder(&ctx);
  auto type = FeltType::get(&ctx, BB_FIELD);
  auto negative = builder.getIndexAttr(-1);
  auto prime = builder.getIndexAttr(BB_PRIME);
  auto abovePrime = builder.getIndexAttr(BB_PRIME + 1);
  expectValue(foldToFelt(builder.getIndexAttr(0), type), 0);
  expectValue(foldToFelt(builder.getIndexAttr(BB_PRIME - 1), type), BB_PRIME - 1);
  expectNoFold(foldToFelt(negative, type));
  expectNoFold(foldToFelt(prime, type));
  expectValue(foldToFelt(negative, type, llzk::cast::OverflowSemantics::SATURATE), 0);
  expectValue(foldToFelt(abovePrime, type, llzk::cast::OverflowSemantics::SATURATE), BB_PRIME - 1);
  expectValue(foldToFelt(negative, type, llzk::cast::OverflowSemantics::WRAP), BB_PRIME - 1);
  expectValue(foldToFelt(prime, type, llzk::cast::OverflowSemantics::WRAP), 0);
  expectValue(foldToFelt(abovePrime, type, llzk::cast::OverflowSemantics::WRAP), 1);
  expectValue(
      foldToFelt(
          builder.getIndexAttr(std::numeric_limits<int64_t>::min()), type,
          llzk::cast::OverflowSemantics::WRAP
      ),
      420548879
  );
  expectValue(
      foldToFelt(
          builder.getIndexAttr(std::numeric_limits<int64_t>::max()), type,
          llzk::cast::OverflowSemantics::WRAP
      ),
      1592717041
  );
  expectNoFold(foldToFelt(negative, type, llzk::cast::OverflowSemantics::TRUNCATE));
  expectNoFold(foldToFelt(prime, type, llzk::cast::OverflowSemantics::TRUNCATE));
  expectNoFold(foldToFelt(abovePrime, type, llzk::cast::OverflowSemantics::TRUNCATE));
  expectValue(
      foldToFelt(builder.getIndexAttr(1LL << 31), type, llzk::cast::OverflowSemantics::TRUNCATE), 0
  );
  expectValue(
      foldToFelt(
          builder.getIndexAttr((1LL << 31) + 1), type, llzk::cast::OverflowSemantics::TRUNCATE
      ),
      1
  );
}

TEST_F(BabyBearFoldTest, CastToFeltCreateOrFoldMaterializesOnlyRepresentableResults) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto type = FeltType::get(&ctx, "goldilocks");
  auto seven = arith::ConstantIndexOp::create(builder, loc, 7);
  Value folded = builder.createOrFold<llzk::cast::IntToFeltOp>(
      loc, type, llzk::cast::OverflowSemantics::ASSERT, seven.getResult()
  );
  auto constant = folded.getDefiningOp<FeltConstantOp>();
  ASSERT_TRUE(constant);
  EXPECT_EQ(constant.getType(), type);
  EXPECT_EQ(constant.getValue().getRawValue(), 7);

  auto negative = arith::ConstantIndexOp::create(builder, loc, -1);
  for (auto mode :
       {llzk::cast::OverflowSemantics::ASSERT, llzk::cast::OverflowSemantics::TRUNCATE}) {
    Value retained = builder.createOrFold<llzk::cast::IntToFeltOp>(
        loc, FeltType::get(&ctx, BB_FIELD), mode, negative.getResult()
    );
    EXPECT_TRUE(retained.getDefiningOp<llzk::cast::IntToFeltOp>());
  }
}

// Exercise the fold hook directly so constant canonicalization cannot mask regressions.
TEST_F(BabyBearFoldTest, UnaryFoldsReduceUncanonicalizedOperands) {
  auto zero = babyBearConst(BB_PRIME);
  auto one = babyBearConst(BB_PRIME + 1);
  expectNoFold(foldUnary<InvFeltOp>(zero));
  expectValue(foldUnary<InvFeltOp>(one), 1);
  expectValue(foldUnary<NegFeltOp>(zero), 0);
  expectValue(foldUnary<NegFeltOp>(one), BB_PRIME - 1);
  expectValue(foldUnary<NotFeltOp>(zero), 134217726);
  expectValue(foldUnary<NotFeltOp>(one), 134217725);
}

TEST_F(BabyBearFoldTest, SignedLiteralIsReducedBeforeBitwiseOperation) {
  auto negative = FeltConstAttr::get(&ctx, llvm::DynamicAPInt(-1), BB_FIELD);
  auto result = foldBinary<AndFeltOp>(negative, babyBearConst(1));
  ASSERT_TRUE(result);
  EXPECT_EQ(result.getRawValue(), llvm::DynamicAPInt(0));
}

TEST_F(BabyBearFoldTest, ShlHugeCountUsesModularExponentiation) {
  // Fermat's little theorem: 2^(p-1) mod p = 1.
  expectValue(foldBinary<ShlFeltOp>(babyBearConst(42), babyBearConst(BB_PRIME - 1)), 42);
}

TEST_F(BabyBearFoldTest, InverseOfZeroDoesNotFold) {
  expectNoFold(foldUnary<InvFeltOp>(babyBearConst(0)));
  expectNoFold(foldUnary<InvFeltOp>(babyBearConst(BB_PRIME)));
}
