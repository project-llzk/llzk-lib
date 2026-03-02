//===-- MemberOverwriteAnalysis.h -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/Analysis/DataFlow/DenseAnalysis.h>

#include <llvm/Support/Debug.h>

namespace llzk {

class MemberOverwriteLattice : public mlir::dataflow::AbstractDenseLattice {
  llvm::DenseMap<llvm::StringRef, component::MemberWriteOp> firstWrites;
  llvm::SetVector<std::pair<component::MemberWriteOp, component::MemberWriteOp>> overwrites;

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  mlir::ChangeResult join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const MemberOverwriteLattice &other) const {
    return std::tie(firstWrites, overwrites) == std::tie(other.firstWrites, other.overwrites);
  }

  void print(llvm::raw_ostream &os) const override;

  mlir::ChangeResult record(component::MemberWriteOp write) {
    auto name = write.getMemberName();

    if (firstWrites.contains(name)) {
      // .insert(...) returns true if an insertion was performed (i.e., it wasn't present before),
      // meaning there was a change
      return mlir::ChangeResult {overwrites.insert({firstWrites.at(name), write})};
    }

    firstWrites.insert({name, write});
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
