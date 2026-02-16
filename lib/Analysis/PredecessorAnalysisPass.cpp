//===-- PredecessorPrinterPass.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ` -llzk-print-predecessors` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/Support/ErrorHandling.h>

using namespace mlir;

namespace llzk {

using namespace function;

#define GEN_PASS_DECL_PREDECESSORPRINTERPASS
#define GEN_PASS_DEF_PREDECESSORPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

/// Prints op without region.
raw_ostream &printRegionless(raw_ostream &os, Operation *op, bool withParent = false) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  if (withParent) {
    if (auto fnOp = op->getParentOfType<FuncDefOp>()) {
      os << '<' << fnOp.getFullyQualifiedName() << ">:";
    } else {
      os << "<(no parent function op)>:";
    }
  }
  op->print(ss, mlir::OpPrintingFlags().skipRegions());
  ss.flush();
  // Skipping regions inserts a new line we don't want, so trim it here.
  llvm::StringRef r(s);
  os << r.rtrim();
  return os;
}

class PredecessorLattice : public mlir::dataflow::AbstractDenseLattice {
  /// Maps op -> [predecessor program points]
  llvm::MapVector<Operation *, llvm::SmallSetVector<Operation *, 4>> predecessors;

public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult visit(Operation *op, Operation *pred) {
    bool newlyInserted = predecessors[op].insert(pred);
    return newlyInserted ? ChangeResult::Change : ChangeResult::NoChange;
  }

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto *other = dynamic_cast<const PredecessorLattice *>(&rhs);
    if (!other) {
      llvm::report_fatal_error("wrong lattice type provided for join");
    }
    ChangeResult r = ChangeResult::NoChange;
    for (auto &[op, preds] : other->predecessors) {
      for (auto pred : preds) {
        r |= visit(op, pred);
      }
    }
    return r;
  }

  ChangeResult meet(const AbstractDenseLattice & /*rhs*/) override {
    llvm::report_fatal_error("meet operation is not supported for PredecessorLattice");
    return ChangeResult::NoChange;
  }

  void print(raw_ostream &os) const override {
    if (predecessors.empty()) {
      os << "(empty)\n";
      return;
    }
    for (auto &[k, v] : predecessors) {
      os.indent(2);
      printRegionless(os, k, true) << " predecessors:";
      llvm::interleave(v, [&os](Operation *p) {
        os << '\n';
        printRegionless(os.indent(6), p, true);
      }, []() {});
      os << '\n';
    }
  }
};

class PredecessorAnalysis
    : public mlir::dataflow::DenseForwardDataFlowAnalysis<PredecessorLattice> {
  /// Stream for debug printing. Currently unused, but propagated from `PredecessorPrinterPass`
  /// in case it is needed in the future.
  [[maybe_unused]]
  raw_ostream &os;

  ProgramPoint *getPoint(const PredecessorLattice &l) const {
    return dyn_cast<ProgramPoint *>(l.getAnchor());
  }

  ChangeResult
  updateLattice(Operation *op, const PredecessorLattice &before, PredecessorLattice *after) {
    ChangeResult result = after->join(before);
    ProgramPoint *pointBefore = getProgramPointBefore(op);
    auto *predState = getOrCreate<mlir::dataflow::PredecessorState>(pointBefore);
    if (!predState->getKnownPredecessors().empty()) {
      for (Operation *pred : predState->getKnownPredecessors()) {
        result |= after->visit(op, pred);
      }
    } else {
      // Predecessor is just the prior or parent op
      Operation *pred = pointBefore->isBlockStart() ? op->getParentOp() : pointBefore->getPrevOp();
      result |= after->visit(op, pred);
    }
    return result;
  }

public:
  using Base = DenseForwardDataFlowAnalysis<PredecessorLattice>;

  PredecessorAnalysis(DataFlowSolver &s, raw_ostream &ro) : Base(s), os(ro) {}

  LogicalResult visitOperation(
      Operation *op, const PredecessorLattice &before, PredecessorLattice *after
  ) override {
    ChangeResult result = updateLattice(op, before, after);
    propagateIfChanged(after, result);
    return success();
  }

  void visitCallControlFlowTransfer(
      CallOpInterface call, mlir::dataflow::CallControlFlowAction action,
      const PredecessorLattice &before, PredecessorLattice *after
  ) override {
    /// `action == CallControlFlowAction::Enter` indicates that:
    ///   - `before` is the state before the call operation;
    ///   - `after` is the state at the beginning of the callee entry block;
    if (action == mlir::dataflow::CallControlFlowAction::EnterCallee) {
      // We skip updating the incoming lattice for function calls to avoid a
      // non-convergence scenario, as calling a function from other contexts
      // can cause the lattice values to oscillate and constantly change.
      setToEntryState(after);
    }
    /// `action == CallControlFlowAction::Exit` indicates that:
    ///   - `before` is the state at the end of a callee exit block;
    ///   - `after` is the state after the call operation.
    else if (action == mlir::dataflow::CallControlFlowAction::ExitCallee) {
      // Get the argument values of the lattice by getting the state as it would
      // have been for the callsite.
      const PredecessorLattice *beforeCall = getLattice(getProgramPointBefore(call));
      ensure(beforeCall, "could not get prior lattice");
      ChangeResult r = after->join(before);
      // Perform a visit so that we see the call op in our lattice
      r |= updateLattice(call, *beforeCall, after);
      propagateIfChanged(after, r);
    }
    /// `action == CallControlFlowAction::External` indicates that:
    ///   - `before` is the state before the call operation.
    ///   - `after` is the state after the call operation, since there is no callee
    ///      body to enter into.
    else if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
      // For external calls, we propagate what information we already have from
      // before the call to after the call, since the external call won't invalidate
      // any of that information. It also, conservatively, makes no assumptions about
      // external calls and their computation, so CDG edges will not be computed over
      // input arguments to external functions.
      join(after, before);
    }
  }

  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> _regionFrom,
      std::optional<unsigned> _regionTo, const PredecessorLattice &before, PredecessorLattice *after
  ) override {
    // The default implementation is `join(after, before)`, but we want to
    // show the predecessor logic for branch operations as well.
    (void)visitOperation(branch, before, after);
  }

protected:
  void setToEntryState(PredecessorLattice *lattice) override {}
};

class PredecessorPrinterPass : public impl::PredecessorPrinterPassBase<PredecessorPrinterPass> {

public:
  PredecessorPrinterPass() : impl::PredecessorPrinterPassBase<PredecessorPrinterPass>() {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();
    // Note: options like `outputStream` are safe to read here, but not in the
    // pass constructor.
    raw_ostream &os = toStream(outputStream);

    DataFlowSolver solver;
    // Alternatively, call `llzk::dataflow::loadRequiredAnalyses`.
    solver.load<mlir::dataflow::SparseConstantPropagation>();
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    solver.load<PredecessorAnalysis>(os);
    LogicalResult res = solver.initializeAndRun(getOperation());

    if (res.failed()) {
      llvm::report_fatal_error("PredecessorAnalysis failed.");
    }

    getOperation()->walk<WalkOrder::PreOrder>([&](FuncDefOp fnOp) {
      Region &fnBody = fnOp.getFunctionBody();
      if (fnBody.empty()) {
        return WalkResult::skip();
      }

      ProgramPoint *point = solver.getProgramPointAfter(fnBody.back().getTerminator());
      PredecessorLattice *finalLattice = solver.getOrCreateState<PredecessorLattice>(point);

      printRegionless(os, fnOp.getOperation()) << ":\n" << *finalLattice << '\n';

      return WalkResult::skip();
    });
  }
};

std::unique_ptr<mlir::Pass> createPredecessorPrinterPass() {
  return std::make_unique<PredecessorPrinterPass>();
}

} // namespace llzk
