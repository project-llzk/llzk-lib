//===-- Walk.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Visitors.h>

#include <llvm/ADT/STLFunctionalExtras.h>
#include <llvm/ADT/SmallVector.h>

/// Returns whether the MLIR walk rooted at `root` contains any `MatchType` instance.
///
/// Traversal stops at the first instance of type `MatchType`.
template <typename MatchType, typename R> inline static bool walkContains(R &root) {
  return root.walk([](MatchType) { return mlir::WalkResult::interrupt(); }).wasInterrupted();
}

/// Returns whether the MLIR walk rooted at `root` contains a `MatchType` instance
/// satisfying `pred`.
///
/// Traversal stops at the first matching instance.
template <typename MatchType, typename R>
inline static bool walkContains(R &root, llvm::function_ref<bool(MatchType)> pred) {
  return root
      .walk([&pred](MatchType t) {
    return pred(t) ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  }).wasInterrupted();
}

/// Walk operations of type `MatchType` from `root` and collect them in walk order.
template <typename MatchType, typename R>
inline static llvm::SmallVector<MatchType> walkCollect(R &root) {
  llvm::SmallVector<MatchType> collected;
  root.walk([&collected](MatchType op) { collected.push_back(op); });
  return collected;
}

/// Walk operations of type `MatchType` from `root` and collect all operations satisfying `pred` in
/// walk order.
template <typename MatchType, typename R>
inline static llvm::SmallVector<MatchType>
walkCollect(R &root, llvm::function_ref<bool(MatchType)> pred) {
  llvm::SmallVector<MatchType> collected;
  root.walk([&collected, &pred](MatchType op) {
    if (pred(op)) {
      collected.push_back(op);
    }
  });
  return collected;
}

/// Walk operations of type `MatchType` from `root` and collect results of applying `map` to the
/// operations in walk order.
template <typename MatchType, typename R, typename Map>
inline static auto walkCollectMapped(R &root, Map &&map) {
  using MappedType = std::invoke_result_t<Map &, MatchType &>;

  llvm::SmallVector<MappedType> collected;
  root.walk([&collected, &map](MatchType op) { collected.push_back(map(op)); });
  return collected;
}
