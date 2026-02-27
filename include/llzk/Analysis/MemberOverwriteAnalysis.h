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
  llvm::DenseMap<llvm::StringRef, llvm::SetVector<component::MemberWriteOp>> memberWrites;

public:
  using AbstractDenseLattice::AbstractDenseLattice;
  mlir::ChangeResult join(const mlir::dataflow::AbstractDenseLattice &other) override;

  bool operator==(const MemberOverwriteLattice &other) const {
    return memberWrites == other.memberWrites;
  }

  void print(llvm::raw_ostream &os) const override;

  mlir::ChangeResult record(component::MemberWriteOp write) {
    auto name = write.getMemberName();
    if (memberWrites.lookup(name).contains(write)) {
      return mlir::ChangeResult::NoChange;
    }
    if (!memberWrites.contains(name)) {
      memberWrites.insert({name, {}});
    }
    memberWrites[name].insert(write);
    return mlir::ChangeResult::Change;
  }

  bool hasOverwrites() const;
  void emitOverwriteErrors() const;
};

class MemberOverwriteAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<MemberOverwriteLattice> {
public:
  MemberOverwriteAnalysis(mlir::DataFlowSolver &solver) : DenseForwardDataFlowAnalysis {solver} {}
  mlir::LogicalResult visitOperation(
      mlir::Operation *op, const MemberOverwriteLattice &before, MemberOverwriteLattice *after
  ) override;

  void setToEntryState(MemberOverwriteLattice *lattice) override {}
};

}; // namespace llzk
