//===-- DynamicAPIntHelper.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/Compare.h"
#include "llzk/Util/DynamicAPIntHelper.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/raw_ostream.h>

#include <limits>

using namespace llvm;
using namespace std;

static DynamicAPInt po2(const DynamicAPInt &e) {
  // Ensure parameter is not negative and that it can be safely cast to unsigned.
  assert(e >= 0);
  assert(e <= std::numeric_limits<unsigned>::max() /* upcast from unsigned -> int64_t */);
  unsigned shiftAmt = llzk::toAPSInt(e).getZExtValue();
  APSInt p = APSInt::get(1) << shiftAmt;
  return llzk::toDynamicAPInt(p);
}

static DynamicAPInt fromBigEndian(const std::vector<bool> &bits) {
  APSInt rawInt(bits.size(), /* isUnsigned */ false);
  for (unsigned i = 0; i < bits.size(); ++i) {
    rawInt.setBitVal(i, bits[i]);
  }
  return llzk::toDynamicAPInt(rawInt);
}

static DynamicAPInt
binaryBitOp(const DynamicAPInt &lhs, const DynamicAPInt &rhs, function_ref<bool(bool, bool)> fn) {
  DynamicAPInt a = lhs, b = rhs;
  std::vector<bool> bits;
  while (a != 0 || b != 0) {
    // bits are sign extended
    bool abit = a != 0 ? bool(int64_t(a % 2)) : lhs < 0;
    bool bbit = b != 0 ? bool(int64_t(b % 2)) : rhs < 0;
    bits.push_back(fn(abit, bbit));
    a /= 2;
    b /= 2;
  }
  // Insert final sign bit, as the above will ignore 0 sign bits. This is also
  // acceptable when both numbers are signed, as it acts as a sign extension.
  bits.push_back(fn(lhs < 0, rhs < 0));
  return fromBigEndian(bits);
}

namespace llzk {

DynamicAPInt operator&(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  auto fn = [](bool a, bool b) { return a && b; };
  return binaryBitOp(lhs, rhs, fn);
}

DynamicAPInt operator|(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  auto fn = [](bool a, bool b) { return a || b; };
  return binaryBitOp(lhs, rhs, fn);
}

DynamicAPInt operator^(const DynamicAPInt &lhs, const DynamicAPInt &rhs) {
  auto fn = [](bool a, bool b) { return a ^ b; };
  return binaryBitOp(lhs, rhs, fn);
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
  if (i.getBitWidth() <= 64) {
    // Fast path for smaller values, just use the int64_t conversion
    return DynamicAPInt(i.isNegative() ? i.getSExtValue() : static_cast<int64_t>(i.getZExtValue()));
  }

  DynamicAPInt res(0), po2(1);
  // Since LLVM 20 doesn't have a direct APInt to DynamicAPInt constructor, we
  // manually construct the DynamicAPInt from bits of the input.
  // We use the positive representation so our negation works at the end.
  APSInt raw = i < 0 ? -i : i;
  for (unsigned b = 0; b < raw.getActiveBits(); b++) {
    DynamicAPInt bitSet(raw[b]);
    res += (bitSet * po2);
    po2 *= 2;
  }
  if (i.isNegative() && res > 0) {
    res = -res;
  }
  return res;
}

APSInt toAPSInt(const DynamicAPInt &i) {
  if (numeric_limits<int64_t>::min() <= i && i <= numeric_limits<int64_t>::max()) {
    // Fast path for smaller values, just use the int64_t conversion
    return APSInt::get(int64_t(i));
  }

  // Else, convert to string and parse back as an APSInt.
  // This may not be the most efficient implementation, but it is the cleanest
  // due to the lack of direct conversions between DynamicAPInt and APInts.
  std::string repr;
  llvm::raw_string_ostream ss(repr);
  ss << i;

  APSInt res(repr);
  // For consistency, we add a bit and mark these as signed integers, since
  // DynamicAPInts are inherently signed.
  res = res.extend(res.getBitWidth() + 1);
  res.setIsSigned(true);

  return res;
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
  assert((base * result) % mod == 1 && "inverse is incorrect");
  return result;
}

llvm::DynamicAPInt modInversePrime(const DynamicAPInt &f, const DynamicAPInt &p) {
  assert(f != 0 && "0 has no inverse");
  // Fermat: f^(p-2) mod p
  DynamicAPInt exp = p - 2;
  return modExp(f, exp, p);
}

} // namespace llzk
