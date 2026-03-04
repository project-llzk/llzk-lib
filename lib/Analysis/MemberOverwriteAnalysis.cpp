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

#define DEBUG_TYPE "member-overwrite-analysis"

using namespace mlir;
using namespace mlir::dataflow;

namespace llzk {
using namespace component;

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const MemberOverwriteLattice &lat) {
  os << lat.mustWrites;
  return os;
}

ChangeResult MemberOverwriteLattice::record(MemberWriteOp write) {
  auto name = write.getMemberName();

  bool changed = false;

  if (mayWrites.contains(name) && mayWrites[name] != write) {
    // .insert(...) returns true if an insertion was performed (i.e., it wasn't present before),
    // meaning there was a change
    changed |= overwrites.insert({mayWrites.at(name), write});
  } else {
    mayWrites.insert({name, write});
    changed = true;
  }

  changed |= mustWrites.insert(name);
  return ChangeResult {changed};
}

ChangeResult MemberOverwriteLattice::join(const AbstractDenseLattice &other) {
  const auto *rhs = dynamic_cast<const MemberOverwriteLattice *>(&other);
  ensure(rhs, "cannot join incomparable lattices");

  auto get = [](LatticeAnchor anchor) { return *dyn_cast<ProgramPoint *>(anchor); };

  LLVM_DEBUG(
      llvm::dbgs() << "Joining " << get(getAnchor()) << "(" << *this << ") with "
                   << get(rhs->getAnchor()) << "(" << *rhs << ")\n"
  );
  bool changed = false;

  // Union the mayWrites
  for (auto [name, write] : rhs->mayWrites) {
    changed |= (!mayWrites.contains(name) || mayWrites[name] != write);
    mayWrites[name] = write;
  }
  changed |= overwrites.set_union(rhs->overwrites);

  // "Intersect" the mustWrites
  changed |= mustWrites.intersect(rhs->mustWrites);

  return ChangeResult {changed};
}

bool MemberOverwriteLattice::hasOverwrites() const { return overwrites.size() > 0; }

void MemberOverwriteLattice::emitOverwriteErrors() const {
  for (auto [first, over] : overwrites) {
    auto diag = over->emitWarning()
                << "overwriting struct member '" << over.getMemberName() << '\'';
    diag.attachNote(first.getLoc()) << "previously written to here";
    diag.report();
  }
}

void MemberOverwriteLattice::ensureWritten(MemberDefOp memberDef) const {
  if (!mustWrites.contains(memberDef.getSymName())) {
    memberDef->emitWarning() << "member may not be written to";
  }
}

void MemberOverwriteLattice::print(llvm::raw_ostream &os) const { os << *this << '\n'; }

LogicalResult MemberOverwriteAnalysis::visitOperation(
    Operation *op, const MemberOverwriteLattice &before, MemberOverwriteLattice *after
) {
  ChangeResult result = after->join(before);

  LLVM_DEBUG(llvm::dbgs() << "Visiting operation: " << *op << ": " << before << "\n");

  if (auto write = dyn_cast<MemberWriteOp>(op)) {
    result |= after->record(write);
  }

  propagateIfChanged(after, result);
  return success();
}

} // namespace llzk
