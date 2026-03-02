//===-- MemberOverwriteAnalysis.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/MemberOverwriteAnalysis.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

namespace llzk {

using namespace mlir;
using namespace mlir::dataflow;
using namespace component;

ChangeResult MemberOverwriteLattice::join(const AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const MemberOverwriteLattice *>(&other);
  ensure(rhs, "cannot join incomparable lattices");

  bool changed = false;
  for (auto [name, write] : rhs->firstWrites) {
    changed |= (!firstWrites.contains(name) || firstWrites[name] != write);
    firstWrites[name] = write;
  }
  changed |= overwrites.set_union(rhs->overwrites);
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

bool MemberOverwriteLattice::hasOverwrites() const {
  return false;
  // for (const auto &[member, writes] : memberWrites) {
  //   if (writes.size() > 1) {
  //     return true;
  //   }
  // }
  // return false;
}

void MemberOverwriteLattice::emitOverwriteErrors() const {
  print(llvm::dbgs());
  for (auto [first, over] : overwrites) {
    auto diag = over->emitWarning() << "overwriting struct member " << over.getMemberName();
    diag.attachNote(first.getLoc()) << "previously written to here";
    diag.report();
  }
  // for (const auto &[member, writes] : memberWrites) {
  //   // Nothing should be empty but check just in case
  //   if (writes.empty() || writes.size() == 1) {
  //     continue;
  //   }
  //   auto firstWrite = writes[0];
  //   for (size_t i = 1; i < writes.size(); i++) {
  //     auto diag = writes[i]->emitWarning() << "overwriting struct member " << member;
  //     diag.attachNote(firstWrite->getLoc()) << "previously written to here";
  //     diag.report();
  //   }
  // }
}

void MemberOverwriteLattice::print(llvm::raw_ostream &os) const {
  // for (const auto &[member, writes] : memberWrites) {
  //   os << member << ":\n";
  //   for (auto write : writes) {
  //     os << '\t' << write << '\n';
  //   }
  // }
}

LogicalResult MemberOverwriteAnalysis::visitOperation(
    Operation *op, const MemberOverwriteLattice &before, MemberOverwriteLattice *after
) {
  ChangeResult result = after->join(before);

  if (auto write = dyn_cast<MemberWriteOp>(op)) {
    result |= after->record(write);
  }

  propagateIfChanged(after, result);
  return success();
}

} // namespace llzk
