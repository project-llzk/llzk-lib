//===-- StreamHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <llvm/Support/raw_ostream.h>

namespace llzk {

/// Wrapper for `llvm::raw_ostream` that filters out certain characters selected by a function.
class filtered_raw_ostream : public llvm::raw_ostream {
  llvm::raw_ostream &underlyingStream;
  std::function<bool(char)> filterFunc;

  void write_impl(const char *ptr, size_t size) override {
    for (size_t i = 0; i < size; ++i) {
      if (!filterFunc(ptr[i])) {
        underlyingStream << ptr[i];
      }
    }
  }

  uint64_t current_pos() const override { return underlyingStream.tell(); }

public:
  filtered_raw_ostream(llvm::raw_ostream &os, std::function<bool(char)> filter)
      : underlyingStream(os), filterFunc(std::move(filter)) {}

  // Moved to a separate file to avoid a weak vtable error.
  ~filtered_raw_ostream() override;
};

/// Generate a string by calling the given `appendFn` with an `llvm::raw_ostream &` as the
/// first argument followed by the additional `Args` provided (if any).
template <typename Func, typename... Args>
inline std::string buildStringViaCallback(Func &&appendFn, Args &&...args) {
  std::string output;
  llvm::raw_string_ostream oss(output);
  std::invoke(std::forward<Func>(appendFn), oss, std::forward<Args>(args)...);
  return output;
}

/// Generate a string by calling `base.print(llvm::raw_ostream &)` on a stream backed by the
/// returned string.
template <typename T, typename... Args>
inline std::string buildStringViaPrint(const T &base, Args &&...args) {
  return buildStringViaCallback([&](llvm::raw_ostream &ss, const T &b) {
    b.print(ss, std::forward<Args>(args)...);
  }, base);
}

/// Generate a string by using the insertion operator (<<) to append all args to a stream backed by
/// the returned string.
template <typename... Args> inline std::string buildStringViaInsertionOp(Args &&...args) {
  std::string output;
  llvm::raw_string_ostream oss(output);
  (oss << ... << std::forward<Args>(args));
  return output;
}

} // namespace llzk
