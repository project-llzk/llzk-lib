//===-- MemberOverwriteAnalysis.cpp -----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/MemberOverwriteAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"
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

  if (auto it = mayWrites.find(name); it != mayWrites.end() && it->second != write) {
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

  LLVM_DEBUG(
      llvm::dbgs() << "Joining " << *dyn_cast<ProgramPoint *>(getAnchor()) << "(" << *this
                   << ") with " << *dyn_cast<ProgramPoint *>(rhs->getAnchor()) << "(" << *rhs
                   << ")\n"
  );
  bool changed = false;

  // Union the mayWrites
  for (auto [name, write] : rhs->mayWrites) {
    auto it = mayWrites.find(name);
    changed |= it == mayWrites.end() || it->second != write;
    mayWrites[name] = write;
  }
  changed |= overwrites.set_union(rhs->overwrites);

  // "Intersect" the mustWrites
  changed |= mustWrites.intersect(rhs->mustWrites);

  return ChangeResult {changed};
}

bool MemberOverwriteLattice::hasOverwrites() const { return overwrites.size() > 0; }

llvm::SetVector<Overwrite> MemberOverwriteLattice::getOverwrites() const { return overwrites; }

bool MemberOverwriteLattice::checkWritten(component::MemberDefOp memberDef) const {
  return mustWrites.contains(memberDef.getSymName());
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

llvm::FailureOr<std::pair<llvm::SetVector<Overwrite>, FuzzySet>>
analyzeStruct(component::StructDefOp structDef) {
  function::FuncDefOp computeOrProductFunc = structDef.getComputeFuncOp();
  if (!computeOrProductFunc) {
    computeOrProductFunc = structDef.getProductFuncOp();
  }

  DataFlowSolver solver {DataFlowConfig {}.setInterprocedural(false)};
  llzk::dataflow::loadRequiredAnalyses(solver);
  solver.load<MemberOverwriteAnalysis>();
  if (failed(solver.initializeAndRun(computeOrProductFunc))) {
    return llvm::failure();
  }

  auto &funcBody = computeOrProductFunc.getBody();
  if (funcBody.empty()) {
    // If there's nothing, just build a default lattice element (no overwrites, everything is
    // unwritten)
    return {{{}, {}}};
  }

  auto *returnOp = funcBody.back().getTerminator();
  const auto *lattice =
      solver.lookupState<MemberOverwriteLattice>(solver.getProgramPointAfter(returnOp));
  return {{lattice->getOverwrites(), lattice->mustWrites}};
}

} // namespace llzk
