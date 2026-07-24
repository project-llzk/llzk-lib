//===-- LLZKAlgebraizeFeltOpsPass.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-algebraize-felt-ops` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>

#include <deque>
#include <optional>
#include <utility>

namespace llzk {
#define GEN_PASS_DEF_ALGEBRAIZEFELTOPSPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-algebraize-felt-ops"

using namespace mlir;
using namespace llzk;
using namespace llzk::boolean;
using namespace llzk::cast;
using namespace llzk::constrain;
using namespace llzk::felt;
using namespace llzk::function;

namespace {

/// Rewrites the non-native felt ops of one constraint-bearing function into
/// native felt arithmetic over `llzk.nondet` witnesses pinned by
/// `constrain.eq`, with checked bit decompositions as the shared backbone
/// (see the pass description in LLZKTransformationPasses.td).
class ConstrainAlgebraizer {
public:
  explicit ConstrainAlgebraizer(FuncDefOp func) : fn(func), b(func) {}

  LogicalResult run();

private:
  FuncDefOp fn;
  OpBuilder b;

  /// Prime modulus per felt type, at bitWidth(p) + 1 so the sign bit is
  /// always clear.
  llvm::DenseMap<Type, llvm::APInt> primes;

  /// Pass-created constants, pooled per (type, value). They are materialized
  /// in a prelude at the top of the entry block — anchored after
  /// `lastPreludeOp` so emission order is preserved — which makes one
  /// instance dominate every use, including uses in sibling regions.
  llvm::DenseMap<std::pair<Type, llvm::APInt>, Value> constPool;
  Operation *lastPreludeOp = nullptr;

  /// Proven upper bounds (in bits) on values. Bounds must be implied by
  /// already-asserted constraints or by op semantics, because a
  /// decomposition at width `w` re-asserts `value < 2^w`; a wrong bound
  /// would reject honest witnesses. Missing entry means full field width;
  /// the tightest bound wins.
  llvm::DenseMap<Value, unsigned> widthBound;

  /// Cache of checked bit decompositions (low bit first). Rewritten ops also
  /// record their results' bits (boolean by construction), so chains of
  /// bitwise ops only decompose their leaves. Vectors live in `bitsStorage`
  /// so returned refs survive later cache insertions.
  std::deque<SmallVector<Value>> bitsStorage;
  llvm::DenseMap<Value, SmallVector<Value> *> bitsCache;

  enum class BitMix { And, Or, Xor };

  const llvm::APInt &primeOf(Type type);
  unsigned fullWidthOf(Type type) { return primeOf(type).getActiveBits(); }

  void setPreludeInsertion();
  Value feltConst(Location loc, Type type, const llvm::APInt &value);
  Value feltConst(Location loc, Type type, uint64_t value);

  /// The canonical (< p) integer value of a felt or integer constant.
  /// `felt.const` does not enforce canonicality, so reduce mod p here; every
  /// consumer (divisor checks, shift amounts, literal bits, guard limits)
  /// must see the field element, not an unreduced representative.
  std::optional<llvm::APInt> getCanonicalConst(Value v);

  void setBound(Value v, unsigned bits);
  unsigned boundOf(Value v);
  void harvestRangeGuard(Value cmpSide, Value trueSide);

  SmallVector<Value> decomposeBits(Location loc, Value v, unsigned width);
  Value recomposeBits(Location loc, ArrayRef<Value> bits);
  ArrayRef<Value> getBits(Location loc, Value v);
  bool canCacheBits(Type type, size_t numBits) { return numBits < fullWidthOf(type); }
  void cacheResultBits(Value result, SmallVector<Value> bits);
  Value bitAt(Location loc, Type type, ArrayRef<Value> bits, unsigned i);
  Value emitBitSlice(Location loc, Type type, ArrayRef<Value> slice);

  Value emitInverse(Location loc, Value v);

  FailureOr<Value> rewriteBitwiseBinary(Operation *op, BitMix mix);
  FailureOr<Value> rewriteBitwiseNot(NotFeltOp op);
  FailureOr<Value> rewriteShl(ShlFeltOp op);
  FailureOr<Value> rewriteShr(ShrFeltOp op);
  FailureOr<Value> rewriteDivModPow2(Operation *op, bool wantQuotient);
};

const llvm::APInt &ConstrainAlgebraizer::primeOf(Type type) {
  auto [it, inserted] = primes.try_emplace(type);
  if (inserted) {
    FeltType feltType = llvm::cast<FeltType>(type);
    const Field &field = feltType.getField();
    it->second = toAPInt(field.prime(), field.bitWidth());
  }
  return it->second;
}

void ConstrainAlgebraizer::setPreludeInsertion() {
  if (lastPreludeOp) {
    b.setInsertionPointAfter(lastPreludeOp);
  } else {
    b.setInsertionPointToStart(&fn.getBody().front());
  }
}

Value ConstrainAlgebraizer::feltConst(Location loc, Type type, const llvm::APInt &value) {
  llvm::APInt norm = value.zextOrTrunc(primeOf(type).getBitWidth());
  auto [it, inserted] = constPool.try_emplace({type, norm});
  if (inserted) {
    OpBuilder::InsertionGuard guard(b);
    setPreludeInsertion();
    auto attr = FeltConstAttr::get(b.getContext(), norm, llvm::cast<FeltType>(type));
    it->second = b.create<FeltConstantOp>(loc, type, attr).getResult();
    lastPreludeOp = it->second.getDefiningOp();
  }
  return it->second;
}

Value ConstrainAlgebraizer::feltConst(Location loc, Type type, uint64_t value) {
  return feltConst(loc, type, llvm::APInt(primeOf(type).getBitWidth(), value));
}

std::optional<llvm::APInt> ConstrainAlgebraizer::getCanonicalConst(Value v) {
  llvm::APInt value;
  FeltConstAttr feltAttr;
  if (matchPattern(v, m_Constant(&feltAttr))) {
    value = feltAttr.getValue();
  } else if (!matchPattern(v, m_ConstantInt(&value))) {
    return std::nullopt;
  }
  auto feltType = llvm::dyn_cast<FeltType>(v.getType());
  if (!feltType) {
    // Integer constants (guard limits before `cast.tofelt`) are already
    // canonical for any field wide enough to hold them.
    return value;
  }
  const Field &field = feltType.getField();
  return toAPInt(field.reduce(value), field.bitWidth());
}

void ConstrainAlgebraizer::setBound(Value v, unsigned bits) {
  unsigned fullWidth = fullWidthOf(v.getType());
  bits = std::max(1u, std::min(bits, fullWidth));
  if (bits == fullWidth) {
    return;
  }
  auto [it, inserted] = widthBound.try_emplace(v, bits);
  if (!inserted && bits < it->second) {
    it->second = bits;
  }
}

unsigned ConstrainAlgebraizer::boundOf(Value v) {
  if (auto it = widthBound.find(v); it != widthBound.end()) {
    return it->second;
  }
  return fullWidthOf(v.getType());
}

/// Records the bound proven by an asserted range guard
/// `constrain.eq(cast.tofelt(bool.cmp lt/le/gt/ge(x, c)), 1)` — the pattern
/// frontends emit to pin blackbox input widths. Only guards at the top level
/// of the body may be harvested: a guard inside a conditional region holds
/// only on its branch.
void ConstrainAlgebraizer::harvestRangeGuard(Value cmpSide, Value trueSide) {
  if (auto castOp = llvm::dyn_cast_if_present<IntToFeltOp>(cmpSide.getDefiningOp())) {
    cmpSide = castOp.getValue();
  }
  auto cmp = llvm::dyn_cast_if_present<CmpOp>(cmpSide.getDefiningOp());
  if (!cmp) {
    return;
  }
  auto pred = cmp.getPredicate();
  Value bounded;
  Value limitSide;
  if (pred == FeltCmpPredicate::LT || pred == FeltCmpPredicate::LE) {
    bounded = cmp.getLhs();
    limitSide = cmp.getRhs();
  } else if (pred == FeltCmpPredicate::GT || pred == FeltCmpPredicate::GE) {
    bounded = cmp.getRhs();
    limitSide = cmp.getLhs();
  } else {
    return;
  }
  auto trueConst = getCanonicalConst(trueSide);
  if (!trueConst || !trueConst->isOne()) {
    return;
  }
  auto limit = getCanonicalConst(limitSide);
  if (!limit) {
    return;
  }
  bool strict = pred == FeltCmpPredicate::LT || pred == FeltCmpPredicate::GT;
  if (strict && limit->isZero()) {
    return;
  }
  llvm::APInt maxVal = strict ? *limit - 1 : *limit;
  setBound(bounded, maxVal.getActiveBits());
}

/// Bit-decomposes `v` into `width` fresh `llzk.nondet` bits, asserting:
///   1. Booleanity: `b_i * (b_i - 1) == 0` for each bit.
///   2. Weighted sum equality: `sum(b_i * 2^i) == v`.
///   3. Range check: `sum < p`, via a bitwise comparison against `p`'s bits.
/// The range check is only needed at full field width: for narrower widths
/// the sum is at most `2^width - 1 < p` already. Callers must only pass a
/// narrow `width` for values known to be `< 2^width`, since the sum equality
/// then also asserts that bound. Returns the bits with the low bit at 0.
SmallVector<Value> ConstrainAlgebraizer::decomposeBits(Location loc, Value v, unsigned width) {
  Type type = v.getType();
  const llvm::APInt &prime = primeOf(type);
  unsigned fullWidth = prime.getActiveBits();
  assert(width >= 1 && width <= fullWidth && "width must be in [1, bits(prime)]");

  Value zero = feltConst(loc, type, 0);
  Value one = feltConst(loc, type, 1);

  SmallVector<Value> bits;
  bits.reserve(width);
  for (unsigned i = 0; i < width; ++i) {
    Value bit = b.create<NonDetOp>(loc, type).getResult();
    bits.push_back(bit);

    Value bMinus1 = b.create<SubFeltOp>(loc, bit, one);
    Value boolProd = b.create<MulFeltOp>(loc, bit, bMinus1);
    b.create<EmitEqualityOp>(loc, boolProd, zero);
  }

  b.create<EmitEqualityOp>(loc, recomposeBits(loc, bits), v);

  if (width < fullWidth) {
    return bits;
  }

  // Range check: sum < p. Two accumulators, each in {0, 1}:
  //   strictLess — the prefix seen so far is strictly less than p's prefix.
  //   stillEqual — the prefix seen so far equals p's prefix.
  // strictLess + stillEqual == 0 means the prefix already exceeded p; the
  // final `strictLess == 1` assert catches that case.
  Value strictLess = zero;
  Value stillEqual = one;
  for (int i = width - 1; i >= 0; --i) {
    Value complement = b.create<SubFeltOp>(loc, one, bits[i]);
    if (prime[i]) {
      Value delta = b.create<MulFeltOp>(loc, stillEqual, complement);
      strictLess = b.create<AddFeltOp>(loc, strictLess, delta);
      stillEqual = b.create<MulFeltOp>(loc, stillEqual, bits[i]);
    } else {
      stillEqual = b.create<MulFeltOp>(loc, stillEqual, complement);
    }
  }
  b.create<EmitEqualityOp>(loc, strictLess, one);

  return bits;
}

/// Combinator inverse of `decomposeBits`: returns `sum(bits[i] * 2^i)`.
/// Emits no assertions; callers are responsible for any booleanity or range
/// constraints on the input bits. Precondition: `bits` is non-empty.
Value ConstrainAlgebraizer::recomposeBits(Location loc, ArrayRef<Value> bits) {
  assert(!bits.empty() && "recomposeBits requires at least one bit");
  Type type = bits[0].getType();
  llvm::APInt weight(primeOf(type).getBitWidth(), 1);

  Value acc = bits[0];
  weight <<= 1;
  for (unsigned i = 1, e = bits.size(); i < e; ++i) {
    Value term = b.create<MulFeltOp>(loc, bits[i], feltConst(loc, type, weight));
    acc = b.create<AddFeltOp>(loc, acc, term);
    weight <<= 1;
  }
  return acc;
}

ArrayRef<Value> ConstrainAlgebraizer::getBits(Location loc, Value v) {
  auto [it, inserted] = bitsCache.try_emplace(v);
  if (inserted) {
    SmallVector<Value> &bits = bitsStorage.emplace_back();
    if (auto c = getCanonicalConst(v)) {
      // Literal 0/1 bits with no assertions.
      for (unsigned i = 0, w = std::max(1u, c->getActiveBits()); i < w; ++i) {
        bits.push_back(feltConst(loc, v.getType(), (*c)[i] ? 1 : 0));
      }
    } else {
      // Emit the decomposition where the value is defined, not where it is
      // first used: the cached bits may be reused from another region (e.g.
      // a sibling scf.if branch), so they must dominate every potential use.
      OpBuilder::InsertionGuard guard(b);
      if (Operation *def = v.getDefiningOp()) {
        b.setInsertionPointAfter(def);
        bits = decomposeBits(loc, v, boundOf(v));
      } else if (auto arg = llvm::dyn_cast<BlockArgument>(v);
                 arg && arg.getOwner() != &fn.getBody().front()) {
        // A nested-region block argument is only in scope inside its owner
        // block. Function entry arguments, on the other hand, belong in the
        // shared prelude so their decompositions can be reused everywhere.
        b.setInsertionPointToStart(arg.getOwner());
        bits = decomposeBits(loc, v, boundOf(v));
      } else {
        setPreludeInsertion();
        bits = decomposeBits(loc, v, boundOf(v));
        lastPreludeOp = &*std::prev(b.getInsertionPoint());
      }
    }
    it->second = &bits;
  }
  return *it->second;
}

/// Bits may only be cached (reused as a value's decomposition) when their
/// weighted sum is provably < p: any sum of fewer than fullWidth bits is
/// < 2^(fullWidth-1) < p. At full width the sum of mixed bits can exceed p,
/// in which case the bits describe a non-canonical representative.
void ConstrainAlgebraizer::cacheResultBits(Value result, SmallVector<Value> bits) {
  if (!canCacheBits(result.getType(), bits.size())) {
    return;
  }
  setBound(result, bits.size());
  bitsStorage.push_back(std::move(bits));
  bitsCache[result] = &bitsStorage.back();
}

/// Reads bit `i` of a decomposition, treating bits past its width as
/// constant zero.
Value ConstrainAlgebraizer::bitAt(Location loc, Type type, ArrayRef<Value> bits, unsigned i) {
  return i < bits.size() ? bits[i] : feltConst(loc, type, 0);
}

/// Recomposes a slice of a checked decomposition; an empty slice is the
/// constant 0. The slice becomes the result's bits.
Value ConstrainAlgebraizer::emitBitSlice(Location loc, Type type, ArrayRef<Value> slice) {
  if (slice.empty()) {
    return feltConst(loc, type, 0);
  }
  Value result = recomposeBits(loc, slice);
  cacheResultBits(result, SmallVector<Value>(slice));
  return result;
}

/// `v * w == 1` pins `w` to the unique inverse and is unsatisfiable for
/// `v == 0`, matching the dialect's requirement that divisors be non-zero.
Value ConstrainAlgebraizer::emitInverse(Location loc, Value v) {
  Type type = v.getType();
  Value w = b.create<NonDetOp>(loc, type).getResult();
  Value prod = b.create<MulFeltOp>(loc, v, w);
  b.create<EmitEqualityOp>(loc, prod, feltConst(loc, type, 1));
  return w;
}

/// Shared scaffolding for bitwise binary felt ops. Operands may have
/// different decomposition widths; for OR/XOR the shorter side is padded
/// with constant zeros (sound: a width-w decomposition proves the value
/// < 2^w), while AND truncates to the shorter width instead, since the
/// upper bits are all `x_i * 0`.
FailureOr<Value> ConstrainAlgebraizer::rewriteBitwiseBinary(Operation *op, BitMix mix) {
  Location loc = op->getLoc();
  ArrayRef<Value> lb = getBits(loc, op->getOperand(0));
  ArrayRef<Value> rb = getBits(loc, op->getOperand(1));
  unsigned n = mix == BitMix::And ? std::min(lb.size(), rb.size()) : std::max(lb.size(), rb.size());
  Type type = op->getResult(0).getType();
  SmallVector<Value> ob;
  ob.reserve(n);
  for (unsigned i = 0; i < n; ++i) {
    Value av = bitAt(loc, type, lb, i);
    Value bv = bitAt(loc, type, rb, i);
    switch (mix) {
    case BitMix::And:
      // out_i = a_i * b_i
      ob.push_back(b.create<MulFeltOp>(loc, av, bv));
      break;
    case BitMix::Or: {
      // out_i = a_i + b_i - a_i * b_i
      Value sum = b.create<AddFeltOp>(loc, av, bv);
      Value prod = b.create<MulFeltOp>(loc, av, bv);
      ob.push_back(b.create<SubFeltOp>(loc, sum, prod));
      break;
    }
    case BitMix::Xor: {
      // out_i = a_i + b_i - 2 * a_i * b_i
      Value sum = b.create<AddFeltOp>(loc, av, bv);
      Value prod = b.create<MulFeltOp>(loc, av, bv);
      Value twoProd = b.create<AddFeltOp>(loc, prod, prod);
      ob.push_back(b.create<SubFeltOp>(loc, sum, twoProd));
      break;
    }
    }
  }
  Value result = recomposeBits(loc, ob);
  cacheResultBits(result, std::move(ob));
  return result;
}

/// NOT complements every felt bit: `out_i = 1 - a_i` at full field width
/// (the one's-complement of the modulus-width representation).
FailureOr<Value> ConstrainAlgebraizer::rewriteBitwiseNot(NotFeltOp op) {
  Location loc = op.getLoc();
  Type type = op.getType();
  ArrayRef<Value> ab = getBits(loc, op.getOperand());
  unsigned fullWidth = fullWidthOf(type);
  Value one = feltConst(loc, type, 1);
  SmallVector<Value> ob;
  ob.reserve(fullWidth);
  for (unsigned i = 0; i < fullWidth; ++i) {
    ob.push_back(b.create<SubFeltOp>(loc, one, bitAt(loc, type, ab, i)));
  }
  return recomposeBits(loc, ob);
}

/// Lowers `felt.shl(a, s) = a * 2^s mod p`.
///
/// A constant `s` folds to a single multiplication by `2^s mod p`, with the
/// result's bits reused from `a`'s when they are cached and the shifted
/// value provably doesn't wrap.
///
/// Otherwise only `s` is bit-decomposed. Let c_i = 2^(2^i) mod p. Then
///   2^s = prod_i (s_i * (c_i - 1) + 1)  (mod p),
/// and the result is `a * 2^s`. Costs ~2 muls per bit plus one decomposition.
FailureOr<Value> ConstrainAlgebraizer::rewriteShl(ShlFeltOp op) {
  Location loc = op.getLoc();
  Type type = op.getType();
  const llvm::APInt &prime = primeOf(type);
  unsigned fullWidth = prime.getActiveBits();
  unsigned constBits = prime.getBitWidth();
  Value lhs = op.getLhs();

  if (auto shift = getCanonicalConst(op.getRhs())) {
    llvm::APInt pow2s = toExactWidthAPInt(
        modExp(llvm::DynamicAPInt(2), toDynamicAPInt(*shift), toDynamicAPInt(prime)), constBits
    );
    Value result = b.create<MulFeltOp>(loc, lhs, feltConst(loc, type, pow2s));
    unsigned s = shift->getLimitedValue(fullWidth);
    setBound(result, boundOf(lhs) + s);
    // Reuse `a`'s cached bits shifted up by `s` when the product can't wrap
    // mod p, so downstream bitwise ops skip a decomposition.
    if (auto it = bitsCache.find(lhs);
        it != bitsCache.end() && canCacheBits(type, it->second->size() + s)) {
      SmallVector<Value> shifted(s, feltConst(loc, type, 0));
      shifted.append(it->second->begin(), it->second->end());
      cacheResultBits(result, std::move(shifted));
    }
    return result;
  }

  ArrayRef<Value> sBits = getBits(loc, op.getRhs());
  Value one = feltConst(loc, type, 1);
  llvm::DynamicAPInt primeDyn = toDynamicAPInt(prime);
  llvm::DynamicAPInt curPow2(2);

  Value pow2s = one;
  for (unsigned i = 0, e = sBits.size(); i < e; ++i) {
    llvm::APInt cMinus1 = toExactWidthAPInt(curPow2 - 1, constBits);
    Value scaled = b.create<MulFeltOp>(loc, sBits[i], feltConst(loc, type, cMinus1));
    Value factor = b.create<AddFeltOp>(loc, scaled, one);
    pow2s = b.create<MulFeltOp>(loc, pow2s, factor);
    curPow2 = (curPow2 * curPow2) % primeDyn;
  }
  return b.create<MulFeltOp>(loc, lhs, pow2s).getResult();
}

/// Lowers `felt.shr(a, s) = floor(a / 2^s)` on the unsigned integer
/// representative of `a`, treating shifts of `a`'s width or more as 0.
///
/// A constant `s` is just a slice of `a`'s checked decomposition: bits
/// [s..w). Otherwise the low `ceil(log2(w))` bits of `s` drive a barrel
/// shifter over the bits of `a`; any higher bit of `s` being set forces the
/// result to 0 via a multiplicative gate.
FailureOr<Value> ConstrainAlgebraizer::rewriteShr(ShrFeltOp op) {
  Location loc = op.getLoc();
  Type type = op.getType();
  ArrayRef<Value> aBits = getBits(loc, op.getLhs());

  if (auto shift = getCanonicalConst(op.getRhs())) {
    unsigned s = shift->getLimitedValue(aBits.size());
    return emitBitSlice(loc, type, aBits.drop_front(s));
  }

  ArrayRef<Value> sBits = getBits(loc, op.getRhs());
  unsigned lanes = aBits.size();
  unsigned levels = llvm::Log2_32_Ceil(lanes);
  Value zero = feltConst(loc, type, 0);

  // Barrel shifter: at level i, conditionally shift right by 2^i controlled
  // by sBits[i]. Positions past `a`'s decomposition width are zero.
  SmallVector<Value> cur(aBits.begin(), aBits.end());
  for (unsigned i = 0; i < levels && i < sBits.size(); ++i) {
    unsigned shift = 1u << i;
    Value ctrl = sBits[i];
    SmallVector<Value> next(lanes);
    for (unsigned j = 0; j < lanes; ++j) {
      Value neighbor = (j + shift < lanes) ? cur[j + shift] : zero;
      Value diff = b.create<SubFeltOp>(loc, neighbor, cur[j]);
      Value scaled = b.create<MulFeltOp>(loc, ctrl, diff);
      next[j] = b.create<AddFeltOp>(loc, cur[j], scaled);
    }
    cur = std::move(next);
  }

  // If any bit of `s` at index >= levels is set, the shift meets or exceeds
  // the value's width, so gate every result bit to 0.
  if (sBits.size() > levels) {
    Value one = feltConst(loc, type, 1);
    Value inRange = one;
    for (unsigned i = levels, e = sBits.size(); i < e; ++i) {
      Value complement = b.create<SubFeltOp>(loc, one, sBits[i]);
      inRange = b.create<MulFeltOp>(loc, inRange, complement);
    }
    for (unsigned j = 0; j < lanes; ++j) {
      cur[j] = b.create<MulFeltOp>(loc, inRange, cur[j]);
    }
  }

  Value result = recomposeBits(loc, cur);
  cacheResultBits(result, std::move(cur));
  return result;
}

/// For a constant divisor c = 2^s, quotient and remainder are slices of the
/// dividend's checked bit decomposition: bits [s..w) and [0..s). Dynamic and
/// non-power-of-two divisors are rejected: a sound encoding of
/// `a == q*b + r` for dynamic `b` needs a multiprecision product argument to
/// rule out field wraparound (e.g. `b = p-1, a = 0` admits the forged
/// `q = 1, r = 1`).
FailureOr<Value> ConstrainAlgebraizer::rewriteDivModPow2(Operation *op, bool wantQuotient) {
  auto divisor = getCanonicalConst(op->getOperand(1));
  if (!divisor) {
    return op->emitError("algebraization supports only constant divisors");
  }
  if (!divisor->isPowerOf2()) {
    return op->emitError("algebraization supports only power-of-two divisors");
  }
  unsigned s = divisor->logBase2();
  Location loc = op->getLoc();
  Type type = op->getResult(0).getType();
  ArrayRef<Value> bits = getBits(loc, op->getOperand(0));
  // A shift of the value's full width or more leaves no quotient bits.
  unsigned split = std::min<unsigned>(s, bits.size());
  return emitBitSlice(loc, type, wantQuotient ? bits.drop_front(split) : bits.take_front(split));
}

LogicalResult ConstrainAlgebraizer::run() {
  // Range guards may sit anywhere in the top-level body: constraints form
  // one conjunction, so a guard bounds its value wherever the value is used.
  for (Operation &op : fn.getBody().front()) {
    if (auto eq = llvm::dyn_cast<EmitEqualityOp>(&op)) {
      harvestRangeGuard(eq.getLhs(), eq.getRhs());
      harvestRangeGuard(eq.getRhs(), eq.getLhs());
    }
  }

  // One pass in program order: operands are defined before uses, so bounds
  // are propagated before the ops consuming them are rewritten, and a
  // rewritten op's result bound (set by cacheResultBits) feeds later ops.
  // The snapshot keeps the loop off the ops the rewrites create.
  SmallVector<Operation *> ops;
  fn.walk([&](Operation *op) { ops.push_back(op); });

  for (Operation *op : ops) {
    b.setInsertionPoint(op);
    // A null Value marks "nothing to replace".
    FailureOr<Value> replacement =
        llvm::TypeSwitch<Operation *, FailureOr<Value>>(op)
            // Bound propagation for native ops.
            .Case<FeltConstantOp>([&](FeltConstantOp c) -> FailureOr<Value> {
      if (auto v = getCanonicalConst(c.getResult())) {
        setBound(c.getResult(), std::max(1u, v->getActiveBits()));
      }
      return Value();
    })
            .Case<IntToFeltOp>([&](IntToFeltOp c) -> FailureOr<Value> {
      // An N-bit integer operand bounds the felt value by 2^N.
      if (auto intType = llvm::dyn_cast<IntegerType>(c.getValue().getType())) {
        setBound(c.getResult(), intType.getWidth());
      }
      return Value();
    })
            .Case<AddFeltOp>([&](AddFeltOp a) -> FailureOr<Value> {
      // No wrap while the bound stays below full width; the clamp in
      // setBound covers the rest (any canonical felt is < 2^fullWidth).
      setBound(a.getResult(), std::max(boundOf(a.getLhs()), boundOf(a.getRhs())) + 1);
      return Value();
    })
            .Case<MulFeltOp>([&](MulFeltOp m) -> FailureOr<Value> {
      setBound(m.getResult(), boundOf(m.getLhs()) + boundOf(m.getRhs()));
      return Value();
    })
            // Rewrites for non-native ops.
            .Case<AndFeltOp>([&](AndFeltOp o) { return rewriteBitwiseBinary(o, BitMix::And); })
            .Case<OrFeltOp>([&](OrFeltOp o) { return rewriteBitwiseBinary(o, BitMix::Or); })
            .Case<XorFeltOp>([&](XorFeltOp o) { return rewriteBitwiseBinary(o, BitMix::Xor); })
            .Case<NotFeltOp>([&](NotFeltOp o) { return rewriteBitwiseNot(o); })
            .Case<ShlFeltOp>([&](ShlFeltOp o) { return rewriteShl(o); })
            .Case<ShrFeltOp>([&](ShrFeltOp o) { return rewriteShr(o); })
            .Case<InvFeltOp>([&](InvFeltOp o) -> FailureOr<Value> {
      return emitInverse(o.getLoc(), o.getOperand());
    })
            .Case<DivFeltOp>([&](DivFeltOp o) -> FailureOr<Value> {
      // Not the cheaper `b * w == a` hint: that leaves `w` unconstrained
      // when a == b == 0. Inverting `b` keeps the result determined and
      // rejects b == 0.
      Value invRhs = emitInverse(o.getLoc(), o.getRhs());
      return b.create<MulFeltOp>(o.getLoc(), o.getLhs(), invRhs).getResult();
    })
            .Case<UnsignedIntDivFeltOp>([&](UnsignedIntDivFeltOp o) {
      return rewriteDivModPow2(o, /*wantQuotient=*/true);
    })
            .Case<UnsignedModFeltOp>([&](UnsignedModFeltOp o) {
      return rewriteDivModPow2(o, /*wantQuotient=*/false);
    }).Default([](Operation *) -> FailureOr<Value> { return Value(); });
    if (failed(replacement)) {
      return failure();
    }
    if (!*replacement) {
      continue;
    }
    op->getResult(0).replaceAllUsesWith(*replacement);
    op->erase();
  }
  return success();
}

class PassImpl : public llzk::impl::AlgebraizeFeltOpsPassBase<PassImpl> {
  using Base = AlgebraizeFeltOpsPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    auto walkResult = getOperation()->walk([](FuncDefOp fn) -> WalkResult {
      if (!fn.hasAllowConstraintAttr() || fn.getBody().empty()) {
        return WalkResult::advance();
      }
      if (failed(ConstrainAlgebraizer(fn).run())) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace
