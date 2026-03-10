//===-- ArrayTypeHelper.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include <optional>

namespace llzk::array {

/// @brief Helper for converting between linear and multi-dimensional indexing with checks to ensure
/// indices are in range for the ArrayType.
class ArrayIndexGen {
  llvm::ArrayRef<int64_t> shape; // owned by the ArrayType from constructor
  int64_t linearSize;
  llvm::SmallVector<int64_t> strides;

  ArrayIndexGen(ArrayType);

public:
  ~ArrayIndexGen() = default;
  ArrayIndexGen(ArrayIndexGen &&) noexcept = default;
  ArrayIndexGen &operator=(ArrayIndexGen &&) noexcept = default;
  ArrayIndexGen(const ArrayIndexGen &) = delete;
  ArrayIndexGen &operator=(const ArrayIndexGen &) = delete;

  /// Construct new ArrayIndexGen. Will assert if hasStaticShape() is false.
  static ArrayIndexGen from(ArrayType);

  std::optional<llvm::SmallVector<mlir::Value>>
  delinearize(int64_t, mlir::Location, mlir::OpBuilder &) const;

  std::optional<llvm::SmallVector<mlir::Attribute>> delinearize(int64_t, mlir::MLIRContext *) const;

  template <typename InListType> std::optional<int64_t> linearize(InListType multiDimIndex) const;

  // NOTE: If the 'multiDimIndex' is shorter than the array rank (i.e., number of dimensions), they
  // indices are treated as the high-order/front dimensions of the array.
  template <typename InListType>
  std::optional<llvm::SmallVector<mlir::Attribute>> checkAndConvert(InListType multiDimIndex);
};

} // namespace llzk::array
