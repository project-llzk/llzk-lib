//===-- Hash.h - Operation Hashing Utilities --------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Operation.h>

#include <functional>

namespace llzk {

template <typename Op>
concept OpHashable = requires(Op op) { op.getOperation(); };

template <OpHashable Op> struct OpHash {
  size_t operator()(const Op &op) const {
    return std::hash<mlir::Operation *> {}(const_cast<Op &>(op).getOperation());
  }
};

} // namespace llzk
