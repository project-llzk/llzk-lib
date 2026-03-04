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

class FuzzySet {
  llvm::DenseMap<llvm::StringRef, bool> isPresent;
  bool _value_is(llvm::StringRef key, bool present) const {
    return isPresent.contains(key) && isPresent.at(key) == present;
  }
  bool _set_to(llvm::StringRef key, bool present) {
    bool changed = !_value_is(key, present);
    isPresent[key] = present;
    return changed;
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const FuzzySet &set);

public:
  bool contains(llvm::StringRef key) const { return _value_is(key, true); }
  bool doesNotContain(llvm::StringRef key) const { return _value_is(key, false); }

  bool insert(llvm::StringRef key) { return _set_to(key, true); }

  bool remove(llvm::StringRef key) { return _set_to(key, false); }

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
    os << (c ? "" : "x") << key << " ";
  }
  os << "]";
  return os;
}

class MemberOverwriteAnalysis;

class MemberOverwriteLattice : public mlir::dataflow::AbstractDenseLattice {
  llvm::DenseMap<llvm::StringRef, component::MemberWriteOp> mayWrites;
  llvm::SetVector<std::pair<component::MemberWriteOp, component::MemberWriteOp>> overwrites;

  FuzzySet mustWrites;

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MemberOverwriteLattice &lat);

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
  void emitOverwriteErrors() const;
  void ensureWritten(component::MemberDefOp) const;
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
