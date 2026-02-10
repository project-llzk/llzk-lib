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
  explicit SourceRefLatticeValue(ScalarTy s) : Base(s) {}
  explicit SourceRefLatticeValue(SourceRef r) : Base(ScalarTy {r}) {}
  SourceRefLatticeValue() : Base(ScalarTy {}) {}
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
  std::pair<SourceRefLatticeValue, mlir::ChangeResult>
  referenceMember(SymbolLookupResult<component::MemberDefOp> memberRef) const;

  /// @brief Perform an array.extract or array.read operation, depending on how many indices
  /// are provided.
  std::pair<SourceRefLatticeValue, mlir::ChangeResult>
  extract(const std::vector<SourceRefIndex> &indices) const;

protected:
  /// @brief Translate this value using the translation map, assuming this value
  /// is a scalar.
  mlir::ChangeResult translateScalar(const TranslationMap &translation);

  /// @brief Perform a recursive transformation over all elements of this value and
  /// return a new value with the modifications.
  virtual std::pair<SourceRefLatticeValue, mlir::ChangeResult>
  elementwiseTransform(llvm::function_ref<SourceRef(const SourceRef &)> transform) const;
};

/// A lattice for use in dense analysis.
class SourceRefLattice : public mlir::dataflow::AbstractDenseLattice {
public:
  // mlir::Value is used for read-like operations that create references in their results,
  // mlir::Operation* is used for write-like operations that reference values as their destinations
  using ValueTy = llvm::PointerUnion<mlir::Value, mlir::Operation *>;
  using ValueMap = mlir::DenseMap<ValueTy, SourceRefLatticeValue>;
  // Used to lookup MLIR values/operations from a given SourceRef (all values that a ref is
  // referenced by)
  using ValueSet = mlir::DenseSet<ValueTy>;
  using Ref2Val = mlir::DenseMap<SourceRef, mlir::DenseSet<ValueTy>>;
  using AbstractDenseLattice::AbstractDenseLattice;

  /* Static utilities */

  /// If val is the source of other values (i.e., a block argument from the function
  /// args or a constant), create the base reference to the val. Otherwise,
  /// return failure.
  /// Our lattice values must originate from somewhere.
  static mlir::FailureOr<SourceRef> getSourceRef(mlir::Value val);

  /* Required methods */

  /// Maximum upper bound
  mlir::ChangeResult join(const AbstractDenseLattice &rhs) override {
    return setValues(static_cast<const SourceRefLattice &>(rhs).valMap);
  }

  /// Minimum lower bound
  mlir::ChangeResult meet(const AbstractDenseLattice & /*rhs*/) override {
    llvm::report_fatal_error("meet operation is not supported for SourceRefLattice");
    return mlir::ChangeResult::NoChange;
  }

  void print(mlir::raw_ostream &os) const override;

  /* Update utility methods */

  mlir::ChangeResult setValues(const ValueMap &rhs);

  mlir::ChangeResult setValue(ValueTy v, const SourceRefLatticeValue &rhs);

  mlir::ChangeResult setValue(ValueTy v, const SourceRef &ref);

  SourceRefLatticeValue getOrDefault(ValueTy v) const;

  SourceRefLatticeValue getReturnValue(unsigned i) const;

  ValueSet lookupValues(const SourceRef &r) const;

  size_t size() const { return valMap.size(); }

  const ValueMap &getMap() const { return valMap; }

  const Ref2Val &getRef2Val() const { return refMap; }

  ValueMap::iterator begin() { return valMap.begin(); }
  ValueMap::iterator end() { return valMap.end(); }
  ValueMap::const_iterator begin() const { return valMap.begin(); }
  ValueMap::const_iterator end() const { return valMap.end(); }

  friend mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRefLattice &v);

private:
  ValueMap valMap;
  Ref2Val refMap;
};

} // namespace llzk

namespace llvm {
class raw_ostream;

raw_ostream &operator<<(raw_ostream &os, llvm::PointerUnion<mlir::Value, mlir::Operation *> ptr);
} // namespace llvm
