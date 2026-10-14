//===-- StaticLoop.h -------------------------------------------*- C++ -*-===//
// Part of the LLZK Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace llzk::polymorphic {
/// Whether another signed iteration exists after current, for current < upper
/// and a positive step. Unsigned subtraction represents the full mathematical
/// distance even when the signed difference would overflow. If another
/// iteration exists, current + step is strictly below upper and cannot overflow.
inline bool hasNextStaticIteration(int64_t current, int64_t upper, int64_t step) {
  return uint64_t(upper) - uint64_t(current) > uint64_t(step);
}
} // namespace llzk::polymorphic
