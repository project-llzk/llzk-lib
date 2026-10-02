//===-- DynamicAPIntHelper.h ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements helpers for DynamicAPInt operations and conversions
/// that LLVM does not provide.
///
/// Note that of the operators defined, bitwise negation ('~') is not implemented.
/// This is because the definition of this operation requires the number of
/// bits to be defined, which may change with dynamically sized integers.
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Compare.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <climits>
#include <cstddef>
#include <cstdint>

namespace llzk {

/// Parse a signed decimal integer, rejecting malformed input without asserting.
llvm::Expected<llvm::DynamicAPInt> parseDynamicAPInt(llvm::StringRef str);

/// Hash the signed numeric value, independent of its width or arithmetic history.
llvm::hash_code hashDynamicAPInt(const llvm::DynamicAPInt &value);

/// Encode a signed or unsigned integer at exactly `bitWidth`, rejecting overflow.
llvm::Expected<llvm::APInt>
checkedToAPInt(const llvm::DynamicAPInt &value, unsigned bitWidth, bool isSigned);

/// Convert to a native signed integer, rejecting values outside its range.
llvm::Expected<int64_t> checkedToInt64(const llvm::DynamicAPInt &value);

/// Convert to a native unsigned integer, rejecting negative values and overflow.
llvm::Expected<uint64_t> checkedToUInt64(const llvm::DynamicAPInt &value);

/// Bitwise operations use infinite two's-complement sign extension.
llvm::DynamicAPInt operator&(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
llvm::DynamicAPInt operator|(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
llvm::DynamicAPInt operator^(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
/// Exact left shift; negative or unrepresentable counts are fatal programming errors.
llvm::DynamicAPInt operator<<(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
/// Arithmetic right shift; arbitrarily large nonnegative counts saturate to 0 or -1.
llvm::DynamicAPInt operator>>(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);

/// Parse trusted decimal text; invalid text is a fatal programming error.
/// Use parseDynamicAPInt for external input.
llvm::DynamicAPInt toDynamicAPInt(llvm::StringRef str);

llvm::DynamicAPInt toDynamicAPInt(const llvm::APSInt &i);

/// Converts an APInt to a DynamicAPInt, using an unsigned interpretation. For a signed
/// interpretation, use `DynamicAPInt(const APInt &)` directly.
llvm::DynamicAPInt toDynamicAPInt(const llvm::APInt &i);

/// Decode 64-bit parts in LSB order as a two's-complement signed integer.
/// The highest bit of the last part is the sign bit; an empty array represents
/// zero. Reports a fatal error if the part count cannot fit APInt's bit width.
llvm::DynamicAPInt toSignedDynamicAPInt(llvm::ArrayRef<uint64_t> parts);

inline llvm::DynamicAPInt toDynamicAPInt(size_t i) {
  return toDynamicAPInt(llvm::APInt(sizeof(size_t) * CHAR_BIT, llzk::checkedCast<uint64_t>(i)));
}

llvm::APSInt toAPSInt(const llvm::DynamicAPInt &i);

/// Modular exponentiation for a nonnegative exponent and positive modulus.
llvm::DynamicAPInt modExp(
    const llvm::DynamicAPInt &base, const llvm::DynamicAPInt &exp, const llvm::DynamicAPInt &mod
);

llvm::DynamicAPInt modInversePrime(const llvm::DynamicAPInt &f, const llvm::DynamicAPInt &p);

} // namespace llzk
