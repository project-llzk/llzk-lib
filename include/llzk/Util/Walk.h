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

/// Returns whether the MLIR walk rooted at `root` contains a `MatchType` instance
/// satisfying `pred`.
///
/// Traversal stops at the first matching instance.
template <typename MatchType, typename R>
inline static bool walkContainsMatch(R &root, llvm::function_ref<bool(MatchType)> pred) {
  return root
      .walk([&pred](MatchType t) {
    return pred(t) ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  }).wasInterrupted();
}

/// Returns whether the MLIR walk rooted at `root` contains any `MatchType` instance.
///
/// Traversal stops at the first instance of type `MatchType`.
template <typename MatchType, typename R> inline static bool walkContains(R &root) {
  return root.walk([](MatchType) { return mlir::WalkResult::interrupt(); }).wasInterrupted();
}

/// Returns whether the MLIR walk rooted at `root` contains any operation whose
/// concrete type is one of `MatchTypes`.
///
/// Traversal stops at the first matching operation.
template <typename MatchType, typename NextMatchType, typename... MatchTypes, typename R>
inline static bool walkContains(R &root) {
  return root
      .walk([](mlir::Operation *op) {
    return llvm::isa<MatchType, NextMatchType, MatchTypes...>(op) ? mlir::WalkResult::interrupt()
                                                                  : mlir::WalkResult::advance();
  }).wasInterrupted();
}

/// Collect all walked operations of type `MatchType` rooted at `root` into a
/// small vector in walk order.
template <typename MatchType, typename R>
inline static llvm::SmallVector<MatchType> walkCollect(R &root) {
  llvm::SmallVector<MatchType> collected;
  root.walk([&collected](MatchType op) { collected.push_back(op); });
  return collected;
}
