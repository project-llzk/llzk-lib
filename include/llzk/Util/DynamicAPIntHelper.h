//===-- DynamicAPIntHelper.h ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements helper methods for constructing DynamicAPInts.
/// These definitions will be mostly obselete when we upgrade to LLVM 21, which
/// defines a DynamicAPInt constructor from an APInt.
///
/// Note that of the operators defined, bitwise negation ('~') is not implemented.
/// This is because the definition of this operation requires the number of
/// bits to be defined, which may change with dynamically sized integers.
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/SlowDynamicAPInt.h>
#include <llvm/ADT/StringRef.h>

namespace llzk {

llvm::DynamicAPInt operator&(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
llvm::DynamicAPInt operator|(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
llvm::DynamicAPInt operator^(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
llvm::DynamicAPInt operator<<(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);
llvm::DynamicAPInt operator>>(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs);

llvm::DynamicAPInt toDynamicAPInt(llvm::StringRef str);

llvm::DynamicAPInt toDynamicAPInt(const llvm::APSInt &i);

inline llvm::DynamicAPInt toDynamicAPInt(const llvm::APInt &i) {
  return toDynamicAPInt(llvm::APSInt(i));
}

llvm::APSInt toAPSInt(const llvm::DynamicAPInt &i);

/// Converts a DynamicAPInt that is non-negative and fits in `bitWidth` bits into an APInt.
/// Uses `bitWidth+1` so that all field values in `[0, p)` — which satisfy
/// `val < 2^bitWidth` — have a clear sign bit and print as positive decimals.
llvm::APInt toAPInt(const llvm::DynamicAPInt &i, unsigned bitWidth);

llvm::DynamicAPInt modExp(
    const llvm::DynamicAPInt &base, const llvm::DynamicAPInt &exp, const llvm::DynamicAPInt &mod
);

llvm::DynamicAPInt modInversePrime(const llvm::DynamicAPInt &f, const llvm::DynamicAPInt &p);

} // namespace llzk
