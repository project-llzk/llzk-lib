//===-- OverflowSemantics.cpp - Cast overflow helpers -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/Util/OverflowSemantics.h"

#include <llvm/Support/ErrorHandling.h>

using namespace llvm;

namespace llzk::cast {
namespace {

static DynamicAPInt positiveModulo(const DynamicAPInt &value, const DynamicAPInt &modulus) {
  DynamicAPInt result = value % modulus;
  return result < 0 ? result + modulus : result;
}

} // namespace

std::optional<DynamicAPInt>
applyIntToFeltOverflow(const DynamicAPInt &value, const Field &field, OverflowSemantics overflow) {
  switch (overflow) {
  case OverflowSemantics::ASSERT:
    return value < 0 || value >= field.prime() ? std::nullopt : std::optional(value);
  case OverflowSemantics::SATURATE:
    return value < 0 ? field.zero() : (value < field.maxVal() ? value : field.maxVal());
  case OverflowSemantics::WRAP:
    return field.reduce(value);
  case OverflowSemantics::TRUNCATE:
    return positiveModulo(value, DynamicAPInt(1) << DynamicAPInt(field.bitWidth()));
  }
  llvm_unreachable("unknown overflow semantics");
}

std::optional<DynamicAPInt> applyFeltToIndexOverflow(
    const DynamicAPInt &value, unsigned indexBitWidth, OverflowSemantics overflow
) {
  const DynamicAPInt safeUpperBound =
      (DynamicAPInt(1) << DynamicAPInt(indexBitWidth - 1)) - DynamicAPInt(1);
  switch (overflow) {
  case OverflowSemantics::ASSERT:
    return value < 0 || value > safeUpperBound ? std::nullopt : std::optional(value);
  case OverflowSemantics::SATURATE:
    return value < 0 ? DynamicAPInt(0) : (value < safeUpperBound ? value : safeUpperBound);
  case OverflowSemantics::WRAP:
  case OverflowSemantics::TRUNCATE:
    return positiveModulo(value, safeUpperBound + DynamicAPInt(1));
  }
  llvm_unreachable("unknown overflow semantics");
}

} // namespace llzk::cast
