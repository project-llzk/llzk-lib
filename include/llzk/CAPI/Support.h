//===-- Support.h - C API general utilities ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares utility macros for working with the C API from the
// C++ side.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/ValueRange.h>

#include <mlir-c/IR.h>

#define LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(dialect, op, suffix, ...)                               \
  MlirOperation llzk##dialect##_##op##Build##suffix(                                               \
      MlirOpBuilder builder, MlirLocation location, __VA_ARGS__                                    \
  )
#define LLZK_DEFINE_OP_BUILD_METHOD(dialect, op, ...)                                              \
  LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(dialect, op, , __VA_ARGS__)

namespace mlir {
template <typename To> auto unwrap_cast(auto &from) { return cast<To>(unwrap(from)); }
} // namespace mlir

constexpr int DEFAULT_ELTS = 5;

/// Helper for unwrapping the C arguments for the map operands
template <int N = DEFAULT_ELTS> class MapOperandsHelper {
public:
  MapOperandsHelper(intptr_t nMapOperands, MlirValueRange const *mapOperands) {
    // resize allocates elements (needed for operator[] access on line 44)
    storage.resize(nMapOperands);
    ranges.reserve(nMapOperands);

    for (intptr_t i = 0; i < nMapOperands; i++) {
      mlir::SmallVector<mlir::Value, N> &sto = storage[i];
      MlirValueRange ops = mapOperands[i];
      ranges.push_back(mlir::ValueRange(unwrapList(ops.size, ops.values, sto)));
    }
  }

  mlir::ArrayRef<mlir::ValueRange> operator*() const { return ranges; }

private:
  mlir::SmallVector<mlir::SmallVector<mlir::Value, N>, N> storage;
  mlir::SmallVector<mlir::ValueRange, N> ranges;
};
