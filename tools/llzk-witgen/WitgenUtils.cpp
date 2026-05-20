//===-- WitgenUtils.cpp - llzk-witgen shared helpers ------------*- C++ -*-===//
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
#include "llzk/Util/DynamicAPIntHelper.h"

#include <mlir/IR/BuiltinTypes.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/Twine.h>

#include <climits>
#include <limits>
#include <random>

using namespace mlir;

namespace llzk::witgen {

static llvm::Expected<size_t>
dynamicAPIntToSize(const llvm::DynamicAPInt &value, llvm::Twine context) {
  if (value < 0) {
    return makeError(context + " would underflow size_t");
  }
  llvm::APSInt as = llzk::toAPSInt(value);
  if (as.getActiveBits() > std::numeric_limits<size_t>::digits) {
    return makeError(context + " would overflow size_t");
  }
  return llzk::checkedCast<size_t>(as.getZExtValue());
}

llvm::Expected<size_t>
checkedDynamicAPIntToSize(const llvm::DynamicAPInt &value, llvm::StringRef context) {
  return dynamicAPIntToSize(value, context);
}

llvm::Expected<size_t> checkedShapeDimToSize(int64_t dim, llvm::StringRef context) {
  if (ShapedType::isDynamic(dim)) {
    return makeError(llvm::Twine(context) + " requires a static shape");
  }
  if (dim < 0) {
    return makeError(llvm::Twine(context) + " has a negative dimension");
  }
  llvm::DynamicAPInt value(dim);
  return dynamicAPIntToSize(value, llvm::Twine(context) + " dimension");
}

llvm::Expected<size_t>
getStaticShapeElementCount(llvm::ArrayRef<int64_t> shape, llvm::StringRef context) {
  llvm::DynamicAPInt count(1);
  for (int64_t dim : shape) {
    auto dimSize = checkedShapeDimToSize(dim, context);
    if (!dimSize) {
      return dimSize.takeError();
    }
    count *= toDynamicAPInt(*dimSize);
  }
  return dynamicAPIntToSize(count, context);
}

llvm::Expected<size_t> getStaticElementCount(ShapedType type, llvm::StringRef context) {
  if (!type.hasStaticShape()) {
    return makeError(llvm::Twine(context) + " requires a static shape");
  }
  int64_t count = type.getNumElements();
  if (count < 0) {
    return makeError(llvm::Twine(context) + " has an invalid negative element count");
  }
  return dynamicAPIntToSize(llvm::DynamicAPInt(count), context);
}

std::mt19937_64 makeDefaultValueRng(const WitgenOptions &options) {
  if (options.randomSeed) {
    return std::mt19937_64(*options.randomSeed);
  }
  std::random_device rd;
  std::seed_seq seed {rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
  return std::mt19937_64(seed);
}

llvm::DynamicAPInt randomFieldElement(std::mt19937_64 &rng, const Field &field) {
  const llvm::DynamicAPInt prime = field.prime();
  if (prime == 0) {
    return prime;
  }

  // Use rejection sampling to produce a uniform value in [0, prime).
  // Generate a random value with the same bit width as the prime; if it is
  // >= prime, discard and retry. The rejection probability is < 50%, so the
  // expected number of iterations is less than 2.
  const unsigned bitWidth = toAPSInt(prime).getActiveBits();
  const unsigned numWords = (bitWidth + 63U) / 64U;
  std::uniform_int_distribution<uint64_t> wordDist;
  llvm::SmallVector<uint64_t, 4> words(numWords);
  while (true) {
    for (uint64_t &word : words) {
      word = wordDist(rng);
    }
    // APInt truncates the top word to exactly bitWidth bits, keeping the
    // candidate in [0, 2^bitWidth).
    llvm::DynamicAPInt value = toDynamicAPInt(llvm::APInt(bitWidth, words));
    if (value < prime) {
      return field.reduce(value);
    }
  }
}

int64_t randomIndexValue(std::mt19937_64 &rng) {
  return std::uniform_int_distribution<
      int64_t>(std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max())(rng);
}

bool randomBoolValue(std::mt19937_64 &rng) {
  return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
}

} // namespace llzk::witgen
