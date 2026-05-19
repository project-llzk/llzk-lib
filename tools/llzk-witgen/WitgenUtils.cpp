//===-- WitgenUtils.cpp - llzk-witgen shared helpers -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenUtils.h"

#include "Errors.h"

#include "llzk/Util/Compare.h"

#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/Twine.h>

#include <limits>
#include <random>

using namespace mlir;

namespace llzk::witgen {

std::mt19937_64 makeDefaultValueRng(const WitgenOptions &options) {
  if (options.randomSeed) {
    return std::mt19937_64(*options.randomSeed);
  }
  std::random_device rd;
  std::seed_seq seed {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
  return std::mt19937_64(seed);
}

llvm::Expected<size_t> checkedShapeDimToSize(int64_t dim, llvm::StringRef context) {
  if (ShapedType::isDynamic(dim)) {
    return makeError(llvm::Twine(context) + " requires a static shape");
  }
  if (dim < 0) {
    return makeError(llvm::Twine(context) + " has a negative dimension");
  }
  if (!std::in_range<size_t>(dim)) {
    return makeError(llvm::Twine(context) + " dimension does not fit in size_t");
  }
  return llzk::checkedCast<size_t>(dim);
}

llvm::Expected<size_t> checkedAddSize(size_t lhs, size_t rhs, llvm::StringRef context) {
  size_t result = 0;
  if (__builtin_add_overflow(lhs, rhs, &result)) {
    return makeError(llvm::Twine("size overflow while computing ") + context);
  }
  return result;
}

llvm::Expected<size_t> checkedMulSize(size_t lhs, size_t rhs, llvm::StringRef context) {
  size_t result = 0;
  if (__builtin_mul_overflow(lhs, rhs, &result)) {
    return makeError(llvm::Twine("size overflow while computing ") + context);
  }
  return result;
}

llvm::Expected<size_t>
getStaticShapeElementCount(llvm::ArrayRef<int64_t> shape, llvm::StringRef context) {
  size_t count = 1;
  for (int64_t dim : shape) {
    auto dimSize = checkedShapeDimToSize(dim, context);
    if (!dimSize) {
      return dimSize.takeError();
    }
    auto nextCount = checkedMulSize(count, *dimSize, context);
    if (!nextCount) {
      return nextCount.takeError();
    }
    count = *nextCount;
  }
  return count;
}

llvm::Expected<size_t> getStaticElementCount(ShapedType type, llvm::StringRef context) {
  if (!type.hasStaticShape()) {
    return makeError(llvm::Twine(context) + " requires a static shape");
  }
  int64_t count = type.getNumElements();
  if (count < 0) {
    return makeError(llvm::Twine(context) + " has an invalid negative element count");
  }
  if (!std::in_range<size_t>(count)) {
    return makeError(llvm::Twine(context) + " element count does not fit in size_t");
  }
  return llzk::checkedCast<size_t>(count);
}

} // namespace llzk::witgen
