//===-- WitgenUtils.h - llzk-witgen shared helpers --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "Errors.h"
#include "WitgenDriver.h"

#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DynamicAPInt.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Error.h>

#include <cstddef>
#include <random>
#include <utility>

namespace llzk::witgen {

template <typename T, typename U> inline llvm::Expected<T> checkedCast(U u) {
  if (std::in_range<T>(u)) {
    return static_cast<T>(u);
  }
  return makeError("lossy integer conversion");
}

/// Seed an RNG for random/default witness value materialization.
std::mt19937_64 makeDefaultValueRng(const WitgenOptions &options);

/// Draw a uniformly distributed field element in `[0, prime)`.
llvm::DynamicAPInt randomFieldElement(std::mt19937_64 &rng, const Field &field);

/// Draw a uniformly distributed signed index value.
int64_t randomIndexValue(std::mt19937_64 &rng);

/// Draw a uniformly distributed boolean value.
bool randomBoolValue(std::mt19937_64 &rng);

/// Convert one static dimension to `size_t`, rejecting dynamic or invalid sizes.
llvm::Expected<size_t> checkedShapeDimToSize(int64_t dim, llvm::StringRef context);

/// Convert a `DynamicAPInt` into `size_t` after validating its range.
llvm::Expected<size_t>
checkedDynamicAPIntToSize(const llvm::DynamicAPInt &value, llvm::StringRef context);

/// Return the static element count for one shape, rejecting dynamic sizes.
llvm::Expected<size_t>
getStaticShapeElementCount(llvm::ArrayRef<int64_t> shape, llvm::StringRef context);

/// Return the static element count for one shaped type.
llvm::Expected<size_t> getStaticElementCount(mlir::ShapedType type, llvm::StringRef context);

} // namespace llzk::witgen
