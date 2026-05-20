//===-- SourceRefLattice.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AbstractLatticeValue.h"
#include "llzk/Analysis/SourceRef.h"
#include "llzk/Analysis/SparseAnalysis.h"
#include "llzk/Util/ErrorHelper.h"

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>

#include <llvm/ADT/PointerUnion.h>

namespace llzk {

class SourceRefLatticeValue;
using TranslationMap = std::unordered_map<SourceRef, SourceRefLatticeValue, SourceRef::Hash>;

/// @brief A value at a given point of the SourceRefLattice.
class SourceRefLatticeValue
    : public dataflow::AbstractLatticeValue<SourceRefLatticeValue, SourceRefSet> {
  using Base = dataflow::AbstractLatticeValue<SourceRefLatticeValue, SourceRefSet>;
  /// For scalar values.
  using ScalarTy = SourceRefSet;
  /// For arrays of values created by, e.g., the LLZK array.new op. A recursive
  /// definition allows arrays to be constructed of other existing values, which is
  /// how the `array.new` operator works.
  /// - Unique pointers are used as each value must be self contained for the
  /// sake of consistent translations. Copies are explicit.
  /// - This array is flattened, with the dimensions stored in another structure.
  /// This simplifies the construction of multidimensional arrays.
  using ArrayTy = std::vector<std::unique_ptr<SourceRefLatticeValue>>;

public:
  explicit SourceRefLatticeValue(ScalarTy s) : Base(std::move(s)) {}
  explicit SourceRefLatticeValue(SourceRef r) : Base(ScalarTy {std::move(r)}) {}
  SourceRefLatticeValue() : Base(ScalarTy {}) {}
  SourceRefLatticeValue(const SourceRefLatticeValue &) = default;
  SourceRefLatticeValue(SourceRefLatticeValue &&) = default;
  SourceRefLatticeValue &operator=(const SourceRefLatticeValue &) = default;
  SourceRefLatticeValue &operator=(SourceRefLatticeValue &&) = default;
  virtual ~SourceRefLatticeValue() = default;

  // Create an empty array of the given shape.
  explicit SourceRefLatticeValue(mlir::ArrayRef<int64_t> shape) : Base(shape) {}

  const SourceRef &getSingleValue() const {
    ensure(isSingleValue(), "not a single value");
    return *getScalarValue().begin();
  }

  /// @brief Directly insert the ref into this value. If this is a scalar value,
  /// insert the ref into the value's set. If this is an array value, the array
  /// is folded into a single scalar, then the ref is inserted.
  mlir::ChangeResult insert(const SourceRef &rhs);

  /// @brief For the refs contained in this value, translate them given the `translation`
  /// map and return the transformed value.
  std::pair<SourceRefLatticeValue, mlir::ChangeResult>
  translate(const TranslationMap &translation) const;

  /// @brief Add the given `memberRef` to the `SourceRef`s contained within this value.
  /// For example, if `memberRef` is a member reference `@foo` and this value represents `%self`,
  /// the new value will represent `%self[@foo]`.
  /// @param memberRef The member reference into the current value.
  /// @return The new value and a change result indicating if the value is different than the
  /// original value.
  mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
  referenceMember(SymbolLookupResult<component::MemberDefOp> memberRef) const;

  /// @brief Perform an array.extract or array.read operation, depending on how many indices
  /// are provided.
  mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
  extract(const std::vector<SourceRefIndex> &indices) const;

protected:
  /// @brief Translate this value using the translation map, assuming this value
  /// is a scalar.
  mlir::ChangeResult translateScalar(const TranslationMap &translation);

  /// @brief Perform a recursive transformation over all elements of this value and
  /// return a new value with the modifications.
  virtual mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
  elementwiseTransform(
      llvm::function_ref<mlir::FailureOr<SourceRef>(const SourceRef &)> transform
  ) const;
};

/// Sparse SSA-value lattice for SourceRef propagation.
class SourceRefLattice : public dataflow::AbstractSparseLattice {
public:
  using LatticeValue = SourceRefLatticeValue;
  // mlir::Value is used for read-like operations that create references in their results,
  // mlir::Operation* is used for write-like operations that reference values as their destinations
  using ValueTy = llvm::PointerUnion<mlir::Value, mlir::Operation *>;
  using Ref2Val = mlir::DenseMap<SourceRef, mlir::DenseSet<ValueTy>>;

  /* Static utilities */

  /// If val is the source of other values (i.e., a block argument, an allocation-like op result,
  /// or a constant), create the base reference to the val. Otherwise,
  /// return failure.
  /// Our lattice values must originate from somewhere.
  static mlir::FailureOr<SourceRef> getSourceRef(mlir::Value val);
  static SourceRefLatticeValue getDefaultValue(ValueTy v);

  using AbstractSparseLattice::AbstractSparseLattice;

  mlir::ChangeResult join(const AbstractSparseLattice &rhs) override;
  mlir::ChangeResult meet(const AbstractSparseLattice &rhs) override;
  void print(mlir::raw_ostream &os) const override;

  const LatticeValue &getValue() const { return value; }

  mlir::ChangeResult setValue(const LatticeValue &newValue);
  mlir::ChangeResult setValue(const SourceRef &ref);

private:
  LatticeValue value;
};

} // namespace llzk

namespace llvm {
class raw_ostream;

raw_ostream &operator<<(raw_ostream &os, llvm::PointerUnion<mlir::Value, mlir::Operation *> ptr);
} // namespace llvm
