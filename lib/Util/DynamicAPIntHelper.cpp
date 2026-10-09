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
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#include <limits>

using namespace llvm;
using namespace std;

static DynamicAPInt po2(const DynamicAPInt &e) {
  // APInt/APSInt bitwidth is limited to max unsigned bits, so must be strictly
  // less than the max to accommodate for the sign bit
  assert(e >= 0);
  assert(e < std::numeric_limits<unsigned>::max());
  unsigned shiftAmt = llzk::toAPSInt(e).getZExtValue();
  APSInt p(shiftAmt + 1, /* isUnsigned */ true);
  p.setBit(shiftAmt);
  return llzk::toDynamicAPInt(p);
}

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
  auto bits = checkedToAPInt(value, 64, true);
  if (!bits) {
    return bits.takeError();
  }
  return bits->getSExtValue();
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

DynamicAPInt operator<<(const DynamicAPInt &lhs, const DynamicAPInt &rhs) { return lhs * po2(rhs); }

DynamicAPInt operator>>(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  if (lhs >= 0) {
    return lhs / po2(rhs);
  } else {
    // round towards negative infinity
    DynamicAPInt divisor = po2(rhs);
    if (lhs % divisor == 0) {
      return lhs / divisor;
    } else {
      return (lhs - (divisor - 1)) / divisor;
    }
  }
}

DynamicAPInt toDynamicAPInt(StringRef str) {
  APSInt parsedInt(str);
  return toDynamicAPInt(parsedInt);
}

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

APSInt toAPSInt(const DynamicAPInt &i) {
  if (numeric_limits<int64_t>::min() <= i && i <= numeric_limits<int64_t>::max()) {
    // Fast path for smaller values, just use the int64_t conversion
    return APSInt::get(int64_t(i));
  }

  // Else, convert to string and parse back as an APSInt.
  // This may not be the most efficient implementation, but it is the cleanest
  // due to the lack of direct conversions between DynamicAPInt and APInts.
  SmallString<64> repr;
  raw_svector_ostream(repr) << i;

  APSInt res(repr);
  // For consistency, we add a bit and mark these as signed integers, since
  // DynamicAPInts are inherently signed.
  res = res.extend(res.getBitWidth() + 1);
  res.setIsSigned(true);

  return res;
}

APInt toAPInt(const DynamicAPInt &val, unsigned bitWidth) {
  SmallString<64> str;
  raw_svector_ostream(str) << val;
  return APInt(bitWidth + 1, str, 10);
}

APInt toExactWidthAPInt(const DynamicAPInt &val, unsigned bitWidth) {
  SmallString<64> str;
  raw_svector_ostream(str) << val;
  return APInt(bitWidth, str, 10);
}

DynamicAPInt modExp(const DynamicAPInt &base, const DynamicAPInt &exp, const DynamicAPInt &mod) {
  DynamicAPInt result(1);
  DynamicAPInt b = base;
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
  assert((f * result) % p == 1 && "inverse is incorrect");
  return result;
}

} // namespace llzk
