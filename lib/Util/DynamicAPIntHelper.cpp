//===-- DynamicAPIntHelper.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/DynamicAPIntHelper.h"

#include "llzk/Util/Compare.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#include <limits>
#include <utility>

using namespace llvm;
using namespace std;

/// Apply a bitwise operation after extending both signed operands to a common width.
static DynamicAPInt binaryBitOp(
    const DynamicAPInt &lhs, const DynamicAPInt &rhs,
    function_ref<APInt(const APInt &, const APInt &)> fn
) {
  APSInt a = llzk::toAPSInt(lhs), b = llzk::toAPSInt(rhs);
  unsigned width = std::max(a.getBitWidth(), b.getBitWidth());
  return DynamicAPInt(fn(a.sext(width), b.sext(width)));
}

namespace llzk {

DynamicAPInt toSignedDynamicAPInt(ArrayRef<uint64_t> parts) {
  if (parts.empty()) {
    return DynamicAPInt(0);
  }
  if (parts.size() <= std::numeric_limits<unsigned>::max() / 64) {
    return DynamicAPInt(APInt(static_cast<unsigned>(parts.size()) * 64, parts));
  }
  llvm::report_fatal_error("part array exceeds APInt's maximum bit width");
}

Expected<DynamicAPInt> parseDynamicAPInt(StringRef str) {
  StringRef digits = str;
  bool negative = digits.consume_front("-");
  APInt magnitude;
  if (digits.getAsInteger(10, magnitude)) {
    return createStringError(inconvertibleErrorCode(), "expected signed decimal integer");
  }
  DynamicAPInt value = toDynamicAPInt(magnitude);
  return negative ? -value : value;
}

hash_code hashDynamicAPInt(const DynamicAPInt &value) {
  APSInt bits = toAPSInt(value);
  return llvm::hash_value(bits.trunc(bits.getSignificantBits()));
}

Expected<APInt> checkedToAPInt(const DynamicAPInt &value, unsigned bitWidth, bool isSigned) {
  APSInt bits = toAPSInt(value);
  bool fits = bitWidth != 0 &&
              (isSigned ? bits.isSignedIntN(bitWidth) : value >= 0 && bits.isIntN(bitWidth));
  if (!fits) {
    return createStringError(inconvertibleErrorCode(), "integer does not fit requested width");
  }
  return isSigned ? bits.sextOrTrunc(bitWidth) : bits.zextOrTrunc(bitWidth);
}

Expected<int64_t> checkedToInt64(const DynamicAPInt &value) {
  if (value < numeric_limits<int64_t>::min() || value > numeric_limits<int64_t>::max()) {
    return createStringError(inconvertibleErrorCode(), "integer does not fit requested width");
  }
  return static_cast<int64_t>(value);
}

Expected<uint64_t> checkedToUInt64(const DynamicAPInt &value) {
  auto bits = checkedToAPInt(value, 64, false);
  if (!bits) {
    return bits.takeError();
  }
  return bits->getZExtValue();
}

DynamicAPInt operator&(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  return binaryBitOp(lhs, rhs, [](const APInt &a, const APInt &b) { return a & b; });
}

DynamicAPInt operator|(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  return binaryBitOp(lhs, rhs, [](const APInt &a, const APInt &b) { return a | b; });
}

DynamicAPInt operator^(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  return binaryBitOp(lhs, rhs, [](const APInt &a, const APInt &b) { return a ^ b; });
}

DynamicAPInt operator<<(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  APSInt bits = toAPSInt(lhs);
  unsigned width = bits.getSignificantBits();
  auto maxAmount =
      toDynamicAPInt(APSInt::getUnsigned(std::numeric_limits<unsigned>::max() - width));
  if (rhs < 0 || rhs > maxAmount) {
    llvm::report_fatal_error("invalid or unrepresentable left shift");
  }
  if (lhs == 0) {
    return DynamicAPInt(0);
  }
  unsigned amount = checkedCast<unsigned>(cantFail(checkedToUInt64(rhs)));
  return DynamicAPInt(bits.sextOrTrunc(width + amount).shl(amount));
}

DynamicAPInt operator>>(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  if (rhs < 0) {
    llvm::report_fatal_error("negative right shift");
  }
  APSInt bits = toAPSInt(lhs);
  if (rhs >= toDynamicAPInt(APSInt::getUnsigned(bits.getSignificantBits()))) {
    return DynamicAPInt(lhs < 0 ? -1 : 0);
  }
  return DynamicAPInt(bits.ashr(checkedCast<unsigned>(cantFail(checkedToUInt64(rhs)))));
}

DynamicAPInt toDynamicAPInt(StringRef str) { return cantFail(parseDynamicAPInt(str)); }

DynamicAPInt toDynamicAPInt(const APSInt &i) {
  // DynamicAPInt interprets APInt (implicit cast from APSInt for the constructor below) as
  // signed. Extend unsigned APSInts with a 0 sign bit so their positive value is preserved.
  if (i.isUnsigned() && i.isSignBitSet()) {
    return DynamicAPInt(i.zext(i.getBitWidth() + 1));
  } else {
    return DynamicAPInt(i);
  }
}

DynamicAPInt toDynamicAPInt(const APInt &i) {
  if (i.isSignBitSet()) {
    return DynamicAPInt(i.zext(i.getBitWidth() + 1));
  } else {
    return DynamicAPInt(i);
  }
}

/// DynamicAPInt has no direct APInt conversion so convert values outside `int64_t` via string.
///
/// Ensure this function doesn't inline so the fast path avoids its stack and register overhead.
static LLVM_ATTRIBUTE_NOINLINE APSInt toAPSIntSlow(const DynamicAPInt &i) {
  // 254-bit magnitude needs at most 77 decimal digits plus a minus sign; 96 inline characters cover
  // this with headroom for larger intermediates so the pre-defined primes don't require resizing.
  SmallString<96> repr;
  raw_svector_ostream(repr) << i;

  // Use the conservative estimate from `APSInt(StringRef)`. Bound the digit count before
  // multiplying or narrowing to APInt's unsigned bit width.
  size_t digits = repr.size() - (repr.front() == '-');
  constexpr uint64_t maxDigits = ((uint64_t(numeric_limits<unsigned>::max()) - 2) * 19) / 64;
  if (digits > maxDigits) {
    llvm::report_fatal_error("decimal integer exceeds APInt's maximum parse width");
  }
  unsigned parseWidth = static_cast<unsigned>((uint64_t(digits) * 64) / 19 + 2);
  APInt bits(parseWidth, repr, /*radix=*/10);
  // getSignificantBits() includes the sign bit for either sign. Truncation
  // allocates a new APInt, so only do it when it saves a storage word.
  unsigned minWidth = bits.getSignificantBits();
  if (APInt::getNumWords(minWidth) < bits.getNumWords()) {
    bits = bits.trunc(minWidth);
  }
  return APSInt(std::move(bits), /*isUnsigned=*/false);
}

APSInt toAPSInt(const DynamicAPInt &i) {
  if (numeric_limits<int64_t>::min() <= i && i <= numeric_limits<int64_t>::max()) {
    // Fast path for smaller values, just use the int64_t conversion
    return APSInt::get(int64_t(i));
  }
  return toAPSIntSlow(i);
}

DynamicAPInt modExp(const DynamicAPInt &base, const DynamicAPInt &exp, const DynamicAPInt &mod) {
  if (exp < 0 || mod <= 0) {
    llvm::report_fatal_error(
        "modular exponentiation requires a nonnegative exponent and positive modulus"
    );
  }
  DynamicAPInt result = llvm::mod(DynamicAPInt(1), mod);
  DynamicAPInt b = llvm::mod(base, mod);
  DynamicAPInt e = exp;
  DynamicAPInt one(1);

  while (e != 0) {
    if (e % 2 != 0) {
      result = (result * b) % mod;
    }

    b = (b * b) % mod;
    e = e >> one;
  }
  return result;
}

DynamicAPInt modInversePrime(const DynamicAPInt &f, const DynamicAPInt &p) {
  assert(f != 0 && "0 has no inverse");
  // Fermat: f^(p-2) mod p
  DynamicAPInt exp = p - 2;
  DynamicAPInt result = modExp(f, exp, p);
  assert(llvm::mod(f * result, p) == 1 && "inverse is incorrect");
  return result;
}

} // namespace llzk
