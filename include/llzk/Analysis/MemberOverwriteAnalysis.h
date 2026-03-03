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

class MemberOverwriteLattice : public mlir::dataflow::AbstractDenseLattice {
  llvm::DenseMap<llvm::StringRef, component::MemberWriteOp> mayWrites;
  llvm::DenseMap<llvm::StringRef, component::MemberWriteOp> mustWrites;
  llvm::SetVector<std::pair<component::MemberWriteOp, component::MemberWriteOp>> overwrites;

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  mlir::ChangeResult join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const MemberOverwriteLattice &other) const {
    return std::tie(mayWrites, overwrites) == std::tie(other.mayWrites, other.overwrites);
  }

  void print(llvm::raw_ostream &os) const override;

  mlir::ChangeResult record(component::MemberWriteOp write) {
    auto name = write.getMemberName();

    if (mayWrites.contains(name)) {
      // .insert(...) returns true if an insertion was performed (i.e., it wasn't present before),
      // meaning there was a change
      return mlir::ChangeResult {overwrites.insert({mayWrites.at(name), write})};
    }

    mayWrites.insert({name, write});
    return mlir::ChangeResult::Change;
  }

  bool hasOverwrites() const;
  void emitOverwriteErrors() const;
};

class MemberOverwriteAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<MemberOverwriteLattice> {
public:
  using DenseForwardDataFlowAnalysis::DenseForwardDataFlowAnalysis;
  mlir::LogicalResult visitOperation(
      mlir::Operation *op, const MemberOverwriteLattice &before, MemberOverwriteLattice *after
  ) override;

  void setToEntryState(MemberOverwriteLattice *lattice) override {}
};

}; // namespace llzk
