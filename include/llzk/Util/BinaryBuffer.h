//===-- BinaryBuffer.h - Little-endian binary buffer ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/DynamicAPIntHelper.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Endian.h>

#include <climits>
#include <cstdint>
#include <limits>

namespace llzk {

/// Accumulate fixed-width integers and field elements in little-endian order.
class BinaryBuffer {
public:
  void writeU32(uint32_t value) { writeInteger(value); }
  void writeU64(uint64_t value) { writeInteger(value); }

  void writeBytes(llvm::ArrayRef<char> bytes) { buffer_.append(bytes.begin(), bytes.end()); }

  /// Append an unsigned fixed-width field encoding, rejecting overflow and negatives.
  /// On failure the buffer is unchanged.
  llvm::Error writeFieldElement(uint32_t size, const llvm::DynamicAPInt &value) {
    if (size == 0 || size > std::numeric_limits<unsigned>::max() / CHAR_BIT) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(), "invalid field encoding size");
    }
    auto exact = checkedToAPInt(value, size * CHAR_BIT, false);
    if (!exact) {
      return exact.takeError();
    }
    for (uint32_t i = 0; i < size; ++i) {
      buffer_.push_back(static_cast<char>(exact->extractBitsAsZExtValue(8, i * 8)));
    }
    return llvm::Error::success();
  }

  uint64_t size() const { return static_cast<uint64_t>(buffer_.size()); }
  llvm::ArrayRef<char> bytes() const { return buffer_; }

private:
  template <typename T> void writeInteger(T value) {
    char bytes[sizeof(T)];
    llvm::support::endian::write<T, llvm::endianness::little>(bytes, value);
    writeBytes(bytes);
  }

  llvm::SmallVector<char> buffer_;
};

} // namespace llzk
