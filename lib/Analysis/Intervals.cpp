//===-- Intervals.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/Intervals.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/ErrorHelper.h"

#include <llvm/ADT/SmallVector.h>

using namespace mlir;

namespace llzk {

/* UnreducedInterval */

Interval UnreducedInterval::reduce(const Field &field) const {
  if (a > b) {
    return Interval::Empty(field);
  }
  if (width() >= field.prime()) {
    return Interval::Entire(field);
  }
  auto lhs = field.reduce(a), rhs = field.reduce(b);
  if (rhs == lhs) {
    return Interval::Degenerate(field, lhs);
  }

  const auto &half = field.half();
  if (lhs <= rhs) {
    if (lhs < half && rhs < half) {
      return Interval::TypeA(field, lhs, rhs);
    } else if (lhs < half) {
      return Interval::TypeC(field, lhs, rhs);
    } else {
      return Interval::TypeB(field, lhs, rhs);
    }
  } else {
    if (lhs >= half && rhs < half) {
      return Interval::TypeF(field, lhs, rhs);
    } else {
      return Interval::Entire(field);
    }
  }
}

UnreducedInterval UnreducedInterval::intersect(const UnreducedInterval &rhs) const {
  const auto &lhs = *this;
  return UnreducedInterval(std::max(lhs.a, rhs.a), std::min(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::doUnion(const UnreducedInterval &rhs) const {
  const auto &lhs = *this;
  return UnreducedInterval(std::min(lhs.a, rhs.a), std::max(lhs.b, rhs.b));
}

UnreducedInterval UnreducedInterval::computeLTPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  DynamicAPInt bound = rhs.b - 1;
  return UnreducedInterval(a, std::min(b, bound));
}

UnreducedInterval UnreducedInterval::computeLEPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  return UnreducedInterval(a, std::min(b, rhs.b));
}

UnreducedInterval UnreducedInterval::computeGTPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  DynamicAPInt bound = rhs.a + 1;
  return UnreducedInterval(std::max(a, bound), b);
}

UnreducedInterval UnreducedInterval::computeGEPart(const UnreducedInterval &rhs) const {
  if (isEmpty() || rhs.isEmpty()) {
    return *this;
  }
  return UnreducedInterval(std::max(a, rhs.a), b);
}

UnreducedInterval UnreducedInterval::operator-() const {
  if (isEmpty()) {
    return *this;
  }
  return UnreducedInterval(-b, -a);
}

UnreducedInterval operator+(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  DynamicAPInt low = lhs.a + rhs.a, high = lhs.b + rhs.b;
  return UnreducedInterval(low, high);
}

UnreducedInterval operator-(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  return lhs + (-rhs);
}

UnreducedInterval operator*(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  DynamicAPInt v1 = lhs.a * rhs.a;
  DynamicAPInt v2 = lhs.a * rhs.b;
  DynamicAPInt v3 = lhs.b * rhs.a;
  DynamicAPInt v4 = lhs.b * rhs.b;

  auto minVal = std::min({v1, v2, v3, v4});
  auto maxVal = std::max({v1, v2, v3, v4});

  return UnreducedInterval(minVal, maxVal);
}

bool UnreducedInterval::overlaps(const UnreducedInterval &rhs) const {
  return isNotEmpty() && rhs.isNotEmpty() && (b >= rhs.a) && (a <= rhs.b);
}

std::strong_ordering operator<=>(const UnreducedInterval &lhs, const UnreducedInterval &rhs) {
  if ((lhs.a < rhs.a) || ((lhs.a == rhs.a) && (lhs.b < rhs.b))) {
    return std::strong_ordering::less;
  }
  if ((lhs.a > rhs.a) || ((lhs.a == rhs.a) && (lhs.b > rhs.b))) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

DynamicAPInt UnreducedInterval::width() const {
  DynamicAPInt w;
  if (a > b) {
    // This would be reduced to an empty Interval, so the width is just zero.
    w = 0;
  } else {
    // Since the range is inclusive, we add one to the difference to get the true width.
    w = (b - a) + 1;
  }
  ensure(w >= 0, "cannot have negative width");
  return w;
}

/* Interval */

const Field &checkFields(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  return lhs.getField();
}

namespace {

llvm::SmallVector<UnreducedInterval, 2> getUnsignedCanonicalParts(const Interval &iv) {
  const Field &f = iv.getField();
  llvm::SmallVector<UnreducedInterval, 2> parts;
  if (iv.isEmpty()) {
    return parts;
  }
  if (iv.isEntire()) {
    parts.emplace_back(f.zero(), f.maxVal());
    return parts;
  }
  if (iv.isTypeF()) {
    parts.emplace_back(iv.lhs(), f.maxVal());
    parts.emplace_back(f.zero(), iv.rhs());
    return parts;
  }

  parts.emplace_back(iv.lhs(), iv.rhs());
  return parts;
}

llvm::SmallVector<UnreducedInterval, 2> getSignedCanonicalParts(const Interval &iv) {
  const Field &f = iv.getField();
  llvm::SmallVector<UnreducedInterval, 2> parts;
  if (iv.isEmpty()) {
    return parts;
  }
  if (iv.isEntire()) {
    parts.emplace_back(f.half() - f.prime(), f.half() - f.one());
    return parts;
  }
  if (iv.isDegenerate()) {
    DynamicAPInt v = iv.lhs();
    if (v < f.half()) {
      parts.emplace_back(v, v);
    } else {
      parts.emplace_back(v - f.prime(), v - f.prime());
    }
    return parts;
  }
  if (iv.isTypeA()) {
    parts.emplace_back(iv.lhs(), iv.rhs());
    return parts;
  }
  if (iv.isTypeB()) {
    parts.emplace_back(iv.lhs() - f.prime(), iv.rhs() - f.prime());
    return parts;
  }
  if (iv.isTypeC()) {
    parts.emplace_back(f.half() - f.prime(), iv.rhs() - f.prime());
    parts.emplace_back(iv.lhs(), f.half() - f.one());
    return parts;
  }

  ensure(iv.isTypeF(), "expected TypeF interval");
  parts.emplace_back(iv.lhs() - f.prime(), iv.rhs());
  return parts;
}

bool containsZero(const UnreducedInterval &iv) { return iv.getLHS() <= 0 && iv.getRHS() >= 0; }

Interval joinDivisionPiece(
    const Field &f, Interval acc, const llvm::DynamicAPInt &q0, const llvm::DynamicAPInt &q1,
    const llvm::DynamicAPInt &q2, const llvm::DynamicAPInt &q3
) {
  DynamicAPInt minQ = std::min({q0, q1, q2, q3});
  DynamicAPInt maxQ = std::max({q0, q1, q2, q3});
  Interval piece = UnreducedInterval(minQ, maxQ).reduce(f);
  return acc.join(piece);
}

} // namespace

UnreducedInterval Interval::toUnreduced() const {
  if (isEmpty()) {
    // Since ranges are inclusive, empty is encoded as `[a, b]` where `a` > `b`.
    // This matches the definition provided by UnreducedInterval::width().
    return UnreducedInterval(field.get().one(), field.get().zero());
  }
  if (isEntire()) {
    return UnreducedInterval(field.get().zero(), field.get().maxVal());
  }
  return UnreducedInterval(a, b);
}

UnreducedInterval Interval::firstUnreduced() const {
  if (is<Type::TypeF>()) {
    return UnreducedInterval(a - field.get().prime(), b);
  }
  return toUnreduced();
}

UnreducedInterval Interval::secondUnreduced() const {
  ensure(is<Type::TypeA, Type::TypeB, Type::TypeC>(), "unsupported range type");
  return UnreducedInterval(a - field.get().prime(), b - field.get().prime());
}

Interval Interval::join(const Interval &rhs) const {
  const auto &lhs = *this;
  const Field &f = checkFields(lhs, rhs);

  // Trivial cases
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(f);
  }
  if (lhs.isEmpty()) {
    return rhs;
  }
  if (rhs.isEmpty()) {
    return lhs;
  }
  if (lhs.isDegenerate() || rhs.isDegenerate()) {
    return lhs.toUnreduced().doUnion(rhs.toUnreduced()).reduce(f);
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    auto newLhs = std::min(lhs.a, rhs.a);
    auto newRhs = std::max(lhs.b, rhs.b);
    if (newLhs == newRhs) {
      return Interval::Degenerate(f, newLhs);
    }
    return Interval(rhs.ty, f, newLhs, newRhs);
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    auto lhsUnred = lhs.firstUnreduced();
    auto opt1 = rhs.firstUnreduced().doUnion(lhsUnred);
    auto opt2 = rhs.secondUnreduced().doUnion(lhsUnred);
    if (opt1.width() <= opt2.width()) {
      return opt1.reduce(f);
    }
    return opt2.reduce(f);
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().doUnion(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().doUnion(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    return Interval::Entire(f);
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.join(lhs);
  }
  llvm::report_fatal_error("unhandled join case");
  return Interval::Entire(f);
}

Interval Interval::intersect(const Interval &rhs) const {
  const auto &lhs = *this;
  const Field &f = checkFields(lhs, rhs);
  // Trivial cases
  if (lhs == rhs) {
    return lhs;
  }
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isEntire()) {
    return rhs;
  }
  if (rhs.isEntire()) {
    return lhs;
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    // These must not be equal
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate()) {
    return Interval::TypeA(f, lhs.a, lhs.a).intersect(rhs);
  }
  if (rhs.isDegenerate()) {
    return Interval::TypeA(f, rhs.a, rhs.a).intersect(lhs);
  }

  // More complex cases
  if (areOneOf<
          {Type::TypeA, Type::TypeA}, {Type::TypeB, Type::TypeB}, {Type::TypeC, Type::TypeC},
          {Type::TypeA, Type::TypeC}, {Type::TypeB, Type::TypeC}>(lhs, rhs)) {
    auto maxA = std::max(lhs.a, rhs.a);
    auto minB = std::min(lhs.b, rhs.b);
    if (maxA < minB) {
      return Interval(lhs.ty, f, maxA, minB);
    } else if (maxA == minB) {
      return Interval::Degenerate(f, maxA);
    } else {
      return Interval::Empty(f);
    }
  }
  if (areOneOf<{Type::TypeA, Type::TypeB}>(lhs, rhs)) {
    return Interval::Empty(f);
  }
  if (areOneOf<{Type::TypeF, Type::TypeF}, {Type::TypeA, Type::TypeF}>(lhs, rhs)) {
    return lhs.firstUnreduced().intersect(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeB, Type::TypeF}>(lhs, rhs)) {
    return lhs.secondUnreduced().intersect(rhs.firstUnreduced()).reduce(f);
  }
  if (areOneOf<{Type::TypeC, Type::TypeF}>(lhs, rhs)) {
    auto rhsUnred = rhs.firstUnreduced();
    auto opt1 = lhs.firstUnreduced().intersect(rhsUnred).reduce(f);
    auto opt2 = lhs.secondUnreduced().intersect(rhsUnred).reduce(f);
    ensure(!opt1.isEntire() && !opt2.isEntire(), "impossible intersection");
    if (opt1.isEmpty()) {
      return opt2;
    }
    if (opt2.isEmpty()) {
      return opt1;
    }
    return opt1.join(opt2);
  }
  if (areOneOf<
          {Type::TypeB, Type::TypeA}, {Type::TypeC, Type::TypeA}, {Type::TypeC, Type::TypeB},
          {Type::TypeF, Type::TypeA}, {Type::TypeF, Type::TypeB}, {Type::TypeF, Type::TypeC}>(
          lhs, rhs
      )) {
    return rhs.intersect(lhs);
  }
  return Interval::Empty(f);
}

Interval Interval::difference(const Interval &other) const {
  const Field &f = checkFields(*this, other);
  // intersect checks that we're in the same field
  Interval intersection = intersect(other);
  if (intersection.isEmpty()) {
    // There's nothing to remove, so just return this
    return *this;
  }

  // Trivial cases with a non-empty intersection
  if (isDegenerate() || other.isEntire()) {
    return Interval::Empty(f);
  }
  if (isEntire()) {
    // Since we don't support punching arbitrary holes in ranges, we only reduce
    // entire ranges if other is [0, b] or [a, prime - 1]
    if (other.a == f.zero()) {
      return UnreducedInterval(other.b + f.one(), f.maxVal()).reduce(f);
    }
    if (other.b == f.maxVal()) {
      return UnreducedInterval(f.zero(), other.a - f.one()).reduce(f);
    }

    return *this;
  }

  // Non-trivial cases
  // - Internal+internal or external+external cases
  if ((is<Type::TypeA, Type::TypeB, Type::TypeC>() &&
       intersection.is<Type::TypeA, Type::TypeB, Type::TypeC>()) ||
      areOneOf<{Type::TypeF, Type::TypeF}>(*this, intersection)) {
    // The intersection needs to be at the end of the interval, otherwise we would
    // split the interval in two, and we aren't set up to support multiple intervals
    // per value.
    if (a != intersection.a && b != intersection.b) {
      return *this;
    }
    // Otherwise, remove the intersection and reduce
    if (a == intersection.a) {
      return UnreducedInterval(intersection.b + f.one(), b).reduce(f);
    }
    // else b == intersection.b
    return UnreducedInterval(a, intersection.a - f.one()).reduce(f);
  }
  // - Mixed internal/external cases. We flip the comparison
  if (isTypeF()) {
    if (a != intersection.b && b != intersection.a) {
      return *this;
    }
    // Otherwise, remove the intersection and reduce
    if (a == intersection.b) {
      return UnreducedInterval(intersection.a + f.one(), b).reduce(f);
    }
    // else b == intersection.a
    return UnreducedInterval(a, intersection.b - f.one()).reduce(f);
  }

  // In cases we don't know how to handle, we over-approximate and return
  // the original interval.
  return *this;
}

Interval Interval::operator-() const { return (-firstUnreduced()).reduce(field.get()); }

Interval Interval::operator~() const {
  return Interval::Degenerate(field.get(), field.get().one()) - *this;
}

Interval operator+(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEntire()) {
    return rhs;
  }
  if (rhs.isEmpty() || lhs.isEntire()) {
    return lhs;
  }
  return (lhs.firstUnreduced() + rhs.firstUnreduced()).reduce(f);
}

Interval operator-(const Interval &lhs, const Interval &rhs) { return lhs + (-rhs); }

Interval operator*(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  auto zeroInterval = Interval::Degenerate(f, f.zero());
  if (lhs == zeroInterval || rhs == zeroInterval) {
    return zeroInterval;
  }
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isEntire() || rhs.isEntire()) {
    return Interval::Entire(f);
  }

  if (Interval::areOneOf<{Interval::Type::TypeB, Interval::Type::TypeB}>(lhs, rhs)) {
    return (lhs.secondUnreduced() * rhs.secondUnreduced()).reduce(f);
  }
  return (lhs.firstUnreduced() * rhs.firstUnreduced()).reduce(f);
}

FailureOr<Interval> feltDiv(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return success(Interval::Empty(f));
  }
  if (!rhs.isDegenerate() || rhs.lhs() == f.zero()) {
    // Supporting arbitrary divisor intervals would require enumerating every
    // possible divisor, inverting each value, and joining the products, which
    // is too expensive. So, we return a failure in the non-degenerate case
    // and in the divide-by-zero case.
    return failure();
  }
  return success(lhs * Interval::Degenerate(f, f.inv(rhs.lhs())));
}

FailureOr<Interval> unsignedIntDiv(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return success(Interval::Empty(f));
  }

  llvm::SmallVector<UnreducedInterval, 2> lhsParts = getUnsignedCanonicalParts(lhs);
  llvm::SmallVector<UnreducedInterval, 2> rhsParts = getUnsignedCanonicalParts(rhs);
  for (const UnreducedInterval &rhsPart : rhsParts) {
    if (rhsPart.getLHS() == f.zero()) {
      return failure();
    }
  }

  Interval result = Interval::Empty(f);
  for (const UnreducedInterval &lhsPart : lhsParts) {
    for (const UnreducedInterval &rhsPart : rhsParts) {
      result = joinDivisionPiece(
          f, result, lhsPart.getLHS() / rhsPart.getRHS(), lhsPart.getLHS() / rhsPart.getLHS(),
          lhsPart.getRHS() / rhsPart.getRHS(), lhsPart.getRHS() / rhsPart.getLHS()
      );
    }
  }
  return success(result);
}

FailureOr<Interval> signedIntDiv(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return success(Interval::Empty(f));
  }

  llvm::SmallVector<UnreducedInterval, 2> lhsParts = getSignedCanonicalParts(lhs);
  llvm::SmallVector<UnreducedInterval, 2> rhsParts = getSignedCanonicalParts(rhs);
  for (const UnreducedInterval &rhsPart : rhsParts) {
    if (containsZero(rhsPart)) {
      return failure();
    }
  }

  Interval result = Interval::Empty(f);
  for (const UnreducedInterval &lhsPart : lhsParts) {
    for (const UnreducedInterval &rhsPart : rhsParts) {
      result = joinDivisionPiece(
          f, result, lhsPart.getLHS() / rhsPart.getLHS(), lhsPart.getLHS() / rhsPart.getRHS(),
          lhsPart.getRHS() / rhsPart.getLHS(), lhsPart.getRHS() / rhsPart.getRHS()
      );
    }
  }
  return success(result);
}

Interval operator%(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  return UnreducedInterval(f.zero(), rhs.b).reduce(f);
}

Interval operator&(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    return Interval::Degenerate(f, lhs.a & rhs.a);
  } else if (lhs.isDegenerate()) {
    return UnreducedInterval(f.zero(), lhs.a).reduce(f);
  } else if (rhs.isDegenerate()) {
    return UnreducedInterval(f.zero(), rhs.a).reduce(f);
  }
  return Interval::Entire(f);
}

Interval operator|(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  auto zeroInterval = Interval::Degenerate(f, f.zero());
  if (lhs == zeroInterval) {
    return rhs;
  }
  if (rhs == zeroInterval) {
    return lhs;
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    return Interval::Degenerate(f, f.reduce(lhs.a | rhs.a));
  }
  return Interval::Entire(f);
}

Interval operator^(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  auto zeroInterval = Interval::Degenerate(f, f.zero());
  if (lhs == zeroInterval) {
    return rhs;
  }
  if (rhs == zeroInterval) {
    return lhs;
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    return Interval::Degenerate(f, f.reduce(lhs.a ^ rhs.a));
  }
  return Interval::Entire(f);
}

Interval operator<<(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    if (rhs.a > f.bitWidth()) {
      return Interval::Entire(f);
    }

    DynamicAPInt v = lhs.a << rhs.a;
    return UnreducedInterval(v, v).reduce(f);
  }
  return Interval::Entire(f);
}

Interval operator>>(const Interval &lhs, const Interval &rhs) {
  const Field &f = checkFields(lhs, rhs);
  if (lhs.isEmpty() || rhs.isEmpty()) {
    return Interval::Empty(f);
  }
  if (lhs.isDegenerate() && rhs.isDegenerate()) {
    if (rhs.a > f.bitWidth()) {
      return Interval::Degenerate(f, f.zero());
    }

    return Interval::Degenerate(f, lhs.a >> rhs.a);
  }
  return Interval::Entire(f);
}

DynamicAPInt Interval::width() const {
  switch (ty) {
  case Type::Empty:
    return field.get().zero();
  case Type::Degenerate:
    return field.get().one();
  case Type::Entire:
    return field.get().prime();
  default:
    return field.get().reduce(toUnreduced().width());
  }
}

Interval boolAnd(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  ensure(lhs.isBoolean() && rhs.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = rhs.getField();

  if (lhs.isBoolFalse() || rhs.isBoolFalse()) {
    return Interval::False(field);
  }
  if (lhs.isBoolTrue() && rhs.isBoolTrue()) {
    return Interval::True(field);
  }

  return Interval::Boolean(field);
}

Interval boolOr(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  ensure(lhs.isBoolean() && rhs.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = rhs.getField();

  if (lhs.isBoolFalse() && rhs.isBoolFalse()) {
    return Interval::False(field);
  }
  if (lhs.isBoolTrue() || rhs.isBoolTrue()) {
    return Interval::True(field);
  }

  return Interval::Boolean(field);
}

Interval boolXor(const Interval &lhs, const Interval &rhs) {
  ensure(
      lhs.getField() == rhs.getField(), "interval operations across differing fields is unsupported"
  );
  ensure(lhs.isBoolean() && rhs.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = rhs.getField();

  // Xor-ing anything with [0, 1] could still result in either case, so just return
  // the full boolean range.
  if (lhs.isBoolEither() || rhs.isBoolEither()) {
    return Interval::Boolean(lhs.getField());
  }

  if (lhs.isBoolTrue() && rhs.isBoolTrue()) {
    return Interval::False(field);
  }
  if (lhs.isBoolTrue() || rhs.isBoolTrue()) {
    return Interval::True(field);
  }
  if (lhs.isBoolFalse() && rhs.isBoolFalse()) {
    return Interval::False(field);
  }

  return Interval::Boolean(field);
}

Interval boolNot(const Interval &iv) {
  ensure(iv.isBoolean(), "operation only supported for boolean-type intervals");
  const auto &field = iv.getField();

  if (iv.isBoolTrue()) {
    return Interval::False(field);
  }
  if (iv.isBoolFalse()) {
    return Interval::True(field);
  }

  return iv;
}

void Interval::print(mlir::raw_ostream &os) const {
  os << TypeName(ty);
  if (is<Type::Degenerate>()) {
    os << '(' << a << ')';
  } else if (!is<Type::Entire, Type::Empty>()) {
    os << ":[ " << a << ", " << b << " ]";
  }
}

} // namespace llzk
