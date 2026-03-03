//===-- MemberOverwriteAnalysis.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/MemberOverwriteAnalysis.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

using namespace mlir;
using namespace mlir::dataflow;

namespace llzk {
using namespace component;

ChangeResult MemberOverwriteLattice::join(const AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const MemberOverwriteLattice *>(&other);
  ensure(rhs, "cannot join incomparable lattices");

  bool changed = false;
  // Union the mayWrites
  for (auto [name, write] : rhs->mayWrites) {
    changed |= (!mayWrites.contains(name) || mayWrites[name] != write);
    mayWrites[name] = write;
  }
  changed |= overwrites.set_union(rhs->overwrites);

  // Intersect the mustWrites
  for (auto [name, _] : mustWrites) {
    if (!rhs->mustWrites.contains(name)) {
      mustWrites.erase(name); // ???
    }
  }

  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

bool MemberOverwriteLattice::hasOverwrites() const { return overwrites.size() > 0; }

void MemberOverwriteLattice::emitOverwriteErrors() const {
  print(llvm::dbgs());
  for (auto [first, over] : overwrites) {
    auto diag = over->emitWarning()
                << "overwriting struct member '" << over.getMemberName() << '\'';
    diag.attachNote(first.getLoc()) << "previously written to here";
    diag.report();
  }
}

void MemberOverwriteLattice::print(llvm::raw_ostream &os) const {}

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
