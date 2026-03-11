//===-- MemberOverwriteAnalysis.h -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>

#include <llvm/Support/Debug.h>

namespace llzk {

/// @brief Represents a set where the membership predicate can take three values: true, false, and
/// "unknown". This is useful for building a lattice for intersection analysis. Internally, we
/// represent the set as a mapping of elements to "present/not present" Booleans; any element for
/// which this mapping doesn't exist is in the default "don't know" state.
class FuzzySet {

  using Present = bool;

  llvm::DenseMap<llvm::StringRef, Present> isPresent;
  bool _value_is(llvm::StringRef key, Present present) const {
    return isPresent.contains(key) && isPresent.at(key) == present;
  }
  bool _set_to(llvm::StringRef key, Present present) {
    bool changed = !_value_is(key, present);
    isPresent[key] = present;
    return changed;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FuzzySet &set);

public:
  /// @brief `key \in *this` == true (key is definitely present)
  /// @param key
  bool contains(llvm::StringRef key) const { return _value_is(key, true); }

  /// @brief `key \in *this` == false (key is definitely not present)
  /// @param key
  bool doesNotContain(llvm::StringRef key) const { return _value_is(key, false); }

  /// @brief Mark `key` as definitely present
  /// @param key
  bool insert(llvm::StringRef key) { return _set_to(key, true); }

  /// @brief Mark `key` as definitely not present
  /// @param key
  bool remove(llvm::StringRef key) { return _set_to(key, false); }

  /// @brief Perform an intersection in-place. If an element is only known about in one of the two
  /// sets being intersected, keep it as-is; otherwise, intersect it as normal.
  /// @param other
  /// @return `true` if this set was modified, `false` otherwise
  bool intersect(const FuzzySet &other) {
    bool changed = false;

    llvm::DenseSet<llvm::StringRef> allKeys;

    for (auto [key, _] : isPresent) {
      allKeys.insert(key);
    }
    for (auto [key, _] : other.isPresent) {
      allKeys.insert(key);
    }

    for (auto key : allKeys) {
      if (isPresent.contains(key) && other.isPresent.contains(key)) {
        changed |= _set_to(key, isPresent.at(key) && other.isPresent.at(key));
      } else if (other.isPresent.contains(key)) {
        changed |= _set_to(key, other.isPresent.at(key));
      }
    }

    return changed;
  }

  bool operator==(const FuzzySet &other) const = default;
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FuzzySet &set) {
  os << "[ ";
  for (auto [key, c] : set.isPresent) {
    os << (c ? "" : "x") << key << ' ';
  }
  os << ']';
  return os;
}

class MemberOverwriteAnalysis;

using Overwrite = std::pair<component::MemberWriteOp, component::MemberWriteOp>;

llvm::FailureOr<std::pair<llvm::SetVector<Overwrite>, FuzzySet>>
    analyzeStruct(component::StructDefOp);

class MemberOverwriteLattice : public mlir::dataflow::AbstractDenseLattice {
  llvm::DenseMap<llvm::StringRef, component::MemberWriteOp> mayWrites;
  llvm::SetVector<Overwrite> overwrites;

  FuzzySet mustWrites;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MemberOverwriteLattice &lat);
  friend llvm::FailureOr<std::pair<llvm::SetVector<Overwrite>, FuzzySet>>
      analyzeStruct(component::StructDefOp);

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  mlir::ChangeResult join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const MemberOverwriteLattice &other) const {
    return std::tie(mayWrites, overwrites, mustWrites) ==
           std::tie(other.mayWrites, other.overwrites, mustWrites);
  }

  void print(llvm::raw_ostream &os) const override;

  void entry() {
    auto structDef = dyn_cast<mlir::ProgramPoint *>(getAnchor())
                         ->getBlock()
                         ->getParentOp()
                         ->getParentOfType<component::StructDefOp>();
    for (auto memberDef : structDef.getMemberDefs()) {
      mustWrites.remove(memberDef.getSymName());
    }
  }

  mlir::ChangeResult record(component::MemberWriteOp write);

  bool hasOverwrites() const;
  llvm::SetVector<Overwrite> getOverwrites() const;
  bool checkWritten(component::MemberDefOp) const;
};

class MemberOverwriteAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<MemberOverwriteLattice> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;
  mlir::LogicalResult visitOperation(
      mlir::Operation *op, const MemberOverwriteLattice &before, MemberOverwriteLattice *after
  ) override;

  void setToEntryState(MemberOverwriteLattice *lattice) override { lattice->entry(); }
};

}; // namespace llzk
