//===-- AbstractLatticeValue.h ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Debug.h"
#include "llzk/Util/ErrorHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Support/LLVM.h>

#include <llvm/Support/Debug.h>

#include <concepts>
#include <type_traits>
#include <variant>

#define DEBUG_TYPE "llzk-abstract-lattice-value"

namespace llzk::dataflow {

template <typename Val>
concept ScalarLatticeValue =
    // Require default constructable
    std::default_initializable<Val> && requires(Val lhs, Val rhs, mlir::raw_ostream &os) {
      // Require a form of print function
      { os << lhs } -> std::same_as<mlir::raw_ostream &>;
      // Require comparability
      { lhs == rhs } -> std::same_as<bool>;
      // Require the ability to combine two scalar values
      { lhs.join(rhs) } -> std::same_as<Val &>;
    };

template <typename Derived, ScalarLatticeValue ScalarTy> class AbstractLatticeValue {
  friend Derived;

  /// For arrays of values created by, e.g., the LLZK array.new op. A recursive
  /// definition allows arrays to be constructed of other existing values, which is
  /// how the `array.new` operator works.
  /// - Unique pointers are used as each value must be self contained for the
  /// sake of consistent translations. Copies are explicit.
  /// - This array is flattened, with the dimensions stored in another structure.
  /// This simplifies the construction of multidimensional arrays.
  using ArrayTy = std::vector<std::unique_ptr<Derived>>;

  /// @brief Create a new array with the given `shape`. The values are pre-allocated
  /// to empty scalar values.
  static ArrayTy constructArrayTy(const mlir::ArrayRef<int64_t> &shape) {
    size_t totalElem = 1;
    for (auto dim : shape) {
      ensure(!mlir::ShapedType::isDynamic(dim), "Cannot pre-allocate dynamically-sized array");
      totalElem *= dim;
    }
    ArrayTy arr(totalElem);
    for (auto it = arr.begin(); it != arr.end(); it++) {
      *it = std::make_unique<Derived>();
    }
    return arr;
  }

  static inline bool isDynamicArray(const mlir::ArrayRef<int64_t> &shape) {
    return mlir::ShapedType::isDynamicShape(shape);
  }

  explicit AbstractLatticeValue(ScalarTy s)
      : value(s), arrayShape(std::nullopt), isDynamic(false) {}
  AbstractLatticeValue() : AbstractLatticeValue(ScalarTy()) {}
  explicit AbstractLatticeValue(const mlir::ArrayRef<int64_t> shape)
      : arrayShape(shape), isDynamic(isDynamicArray(shape)) {
    if (isDynamic) {
      value = ScalarTy();
    } else {
      value = constructArrayTy(shape);
    }
  }

  AbstractLatticeValue(const AbstractLatticeValue &rhs) { *this = rhs; }
  AbstractLatticeValue(AbstractLatticeValue &&rhs) = default;

  // Enable copying by duplicating unique_ptrs and copying the contained values.
  AbstractLatticeValue &operator=(const AbstractLatticeValue &rhs) {
    copyArrayShape(rhs);
    if (rhs.isScalar() || rhs.isDynamicArray()) {
      getValue() = rhs.getScalarValue();
    } else {
      // create an empty array of the same size
      getValue() = constructArrayTy(rhs.getArrayShape());
      auto &lhsArr = getArrayValue();
      auto &rhsArr = rhs.getArrayValue();
      for (unsigned i = 0; i < lhsArr.size(); i++) {
        // Recursive copy assignment of lattice values
        *lhsArr[i] = *rhsArr[i];
      }
    }
    return *this;
  }
  AbstractLatticeValue &operator=(AbstractLatticeValue &&rhs) = default;

public:
  bool isScalar() const { return std::holds_alternative<ScalarTy>(value); }
  bool isSingleValue() const { return isScalar() && getScalarValue().size() == 1; }
  bool isArray() const { return std::holds_alternative<ArrayTy>(value); }
  bool isDynamicArray() const { return isDynamic; }

  const ScalarTy &getScalarValue() const {
    ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  ScalarTy &getScalarValue() {
    ensure(isScalar(), "not a scalar value");
    return std::get<ScalarTy>(value);
  }

  const ArrayTy &getArrayValue() const {
    ensure(isArray() && !isDynamicArray(), "not a static array value");
    return std::get<ArrayTy>(value);
  }

  ArrayTy &getArrayValue() {
    ensure(isArray() && !isDynamicArray(), "not a static array value");
    return std::get<ArrayTy>(value);
  }

  /// @brief Directly index into the flattened array using a single index.
  const Derived &getElemFlatIdx(size_t i) const {
    ensure(isArray() && !isDynamicArray(), "not a static array value");
    auto &arr = getArrayValue();
    ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  Derived &getElemFlatIdx(size_t i) {
    ensure(isArray() && !isDynamicArray(), "not a static array value");
    auto &arr = getArrayValue();
    ensure(i < arr.size(), "index out of range");
    return *arr.at(i);
  }

  size_t getArraySize() const { return getArrayValue().size(); }

  size_t getNumArrayDims() const { return getArrayShape().size(); }

  void print(mlir::raw_ostream &os) const {
    if (isScalar() || isDynamicArray()) {
      os << getScalarValue();
    } else {
      os << "[ ";
      const auto &arr = getArrayValue();
      for (auto it = arr.begin(); it != arr.end();) {
        (*it)->print(os);
        it++;
        if (it != arr.end()) {
          os << ", ";
        } else {
          os << ' ';
        }
      }
      os << ']';
    }
  }

  /// @brief If this is an array value, combine all elements into a single scalar
  /// value and return it. If this is already a scalar value, return the scalar value.
  ScalarTy foldToScalar() const {
    if (isScalar()) {
      return getScalarValue();
    }

    ScalarTy res;
    for (auto &val : getArrayValue()) {
      auto rhs = val->foldToScalar();
      res.join(rhs);
    }
    return res;
  }

  /// @brief Sets this value to be equal to `rhs`.
  /// @return A `mlir::ChangeResult` indicating if an update was performed or not.
  mlir::ChangeResult setValue(const AbstractLatticeValue &rhs) {
    if (*this == rhs) {
      return mlir::ChangeResult::NoChange;
    }
    *this = rhs;
    return mlir::ChangeResult::Change;
  }

  /// @brief Union this value with that of rhs.
  mlir::ChangeResult update(const Derived &rhs) {
    if (isScalar() && rhs.isScalar()) {
      return updateScalar(rhs.getScalarValue());
    } else if (isArray() && rhs.isArray() && getArraySize() == rhs.getArraySize()) {
      return updateArray(rhs.getArrayValue());
    } else {
      return foldAndUpdate(rhs);
    }
  }

  bool operator==(const AbstractLatticeValue &rhs) const {
    if (isScalar() && rhs.isScalar()) {
      return getScalarValue() == rhs.getScalarValue();
    } else if (isArray() && rhs.isArray() && getArraySize() == rhs.getArraySize()) {
      for (size_t i = 0; i < getArraySize(); i++) {
        if (getElemFlatIdx(i) != rhs.getElemFlatIdx(i)) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

protected:
  std::variant<ScalarTy, ArrayTy> &getValue() { return value; }

  const std::vector<int64_t> &getArrayShape() const {
    ensure(arrayShape != std::nullopt, "not an array value");
    return arrayShape.value();
  }

  int64_t getArrayDim(unsigned i) const {
    const auto &arrShape = getArrayShape();
    ensure(i < arrShape.size(), "dimension index out of bounds");
    return arrShape.at(i);
  }

  void copyArrayShape(const AbstractLatticeValue &rhs) {
    arrayShape = rhs.arrayShape;
    isDynamic = rhs.isDynamic;
  }

  /// @brief Union this value with the given scalar.
  mlir::ChangeResult updateScalar(const ScalarTy &rhs) {
    auto lhs = getScalarValue();
    lhs.join(rhs);
    if (getScalarValue() == lhs) {
      return mlir::ChangeResult::NoChange;
    }
    getScalarValue() = lhs;
    return mlir::ChangeResult::Change;
  }

  /// @brief Union this value with the given array.
  mlir::ChangeResult updateArray(const ArrayTy &rhs) {
    mlir::ChangeResult res = mlir::ChangeResult::NoChange;
    auto &lhs = getArrayValue();
    for (size_t i = 0; i < getArraySize(); i++) {
      res |= lhs[i]->update(*rhs.at(i));
    }
    return res;
  }

  /// @brief Folds the current value into a scalar and folds `rhs` to a scalar and updates
  /// the current value to the union of the two scalars.
  mlir::ChangeResult foldAndUpdate(const Derived &rhs) {
    auto folded = foldToScalar();
    auto rhsScalar = rhs.foldToScalar();
    folded.join(rhsScalar);
    if (isScalar() && getScalarValue() == folded) {
      return mlir::ChangeResult::NoChange;
    }
    getValue() = folded;
    return mlir::ChangeResult::Change;
  }

private:
  std::variant<ScalarTy, ArrayTy> value;
  std::optional<std::vector<int64_t>> arrayShape;
  bool isDynamic;
};

template <typename Derived, ScalarLatticeValue ScalarTy>
mlir::raw_ostream &
operator<<(mlir::raw_ostream &os, const AbstractLatticeValue<Derived, ScalarTy> &v) {
  v.print(os);
  return os;
}

} // namespace llzk::dataflow

#undef DEBUG_TYPE
