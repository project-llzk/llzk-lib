//===-- AlignmentHelper.h --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>

#include <concepts>

namespace llzk::alignmentHelpers {

template <class ValueT, class FnT>
concept Matcher = requires(FnT fn, ValueT val) {
  { fn(val, val) } -> std::convertible_to<bool>;
};

template <class ValueT, class FnT>
  requires Matcher<ValueT, FnT>
llvm::FailureOr<llvm::SetVector<std::pair<ValueT, ValueT>>> getMatchingPairs(
    llvm::ArrayRef<ValueT> as, llvm::ArrayRef<ValueT> bs, FnT doesMatch, bool allowPartial = true
) {

  llvm::SetVector<ValueT> setA {as.begin(), as.end()}, setB {bs.begin(), bs.end()};
  llvm::DenseMap<size_t, llvm::SmallVector<size_t>> possibleMatchesA, possibleMatchesB;

  for (size_t i = 0, ea = as.size(), eb = bs.size(); i < ea; i++) {
    for (size_t j = 0; j < eb; j++) {
      if (doesMatch(as[i], bs[j])) {
        possibleMatchesA[i].push_back(j);
        possibleMatchesB[j].push_back(i);
      }
    }
  }

  llvm::SetVector<std::pair<ValueT, ValueT>> matches;
  for (auto [a, b] : possibleMatchesA) {
    if (b.size() == 1 && possibleMatchesB[b[0]].size() == 1) {
      setA.remove(as[a]);
      setB.remove(bs[b[0]]);
      matches.insert({as[a], bs[b[0]]});
    }
  }

  if ((!setA.empty() || !setB.empty()) && !allowPartial) {
    return llvm::failure();
  }
  return matches;
}
} // namespace llzk::alignmentHelpers
