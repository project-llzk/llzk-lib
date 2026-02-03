//===-- TestAnalysisPass.cpp -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the ` -llzk-test-pass` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <llvm/Support/ErrorHandling.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <llvm/ADT/DenseMap.h>

#include <map>

using namespace mlir;

namespace llzk {

using namespace function;

#define GEN_PASS_DECL_TESTANALYSISPASS
#define GEN_PASS_DEF_TESTANALYSISPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

raw_ostream &printOp(raw_ostream &os, Operation *op) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  op->print(ss, mlir::OpPrintingFlags().skipRegions());
  ss.flush();
  // Skipping regions inserts a new line we don't want, so trim it here.
  llvm::StringRef r(s);
  os << r.rtrim();
  return os;
}

class TestLattice : public mlir::dataflow::AbstractDenseLattice {
  std::map<size_t, Operation*> op_traversal;

public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult visit(Operation *op, size_t order) {
    if (op_traversal.contains(order)) {
      return ChangeResult::NoChange;
    }
    op_traversal[order] = op;
    return ChangeResult::Change;
  }

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const TestLattice &>(rhs);
    ChangeResult r = ChangeResult::NoChange;
    for (auto &[k, v] : other.op_traversal) {
      if (!op_traversal.contains(k)) {
        op_traversal[k] = v;
        r |= ChangeResult::Change;
      }
    }
    return r;
  }

  ChangeResult meet(const AbstractDenseLattice & /*rhs*/) override {
    llvm::report_fatal_error("meet operation is not supported for TestLattice");
    return ChangeResult::NoChange;
  }

  void print(raw_ostream &os) const override {
    if (op_traversal.empty()) {
      os << "(empty)\n";
      return;
    }
    for (auto &[k, v] : op_traversal) {
      os.indent(2);
      os << k << ": ";
      printOp(os, v) <<  '\n';
    }
  }
};

class TestAnalysis : public mlir::dataflow::DenseForwardDataFlowAnalysis<TestLattice> {
  /// Tracks how many ops the analysis has traversed.
  size_t traversed;

  /// Stream for debug printing. Currently unused, but propagated from `TestAnalysisPass`
  /// in case it is needed in the future.
  [[maybe_unused]]
  raw_ostream &os;
public:
  using Base = DenseForwardDataFlowAnalysis<TestLattice>;

  TestAnalysis(DataFlowSolver &solver, raw_ostream &ro) : Base(solver), os(ro) {}

  LogicalResult visitOperation(Operation *op, const TestLattice &before, TestLattice *after) override {
    ChangeResult result = after->join(before);
    result |= after->visit(op, traversed++);
    propagateIfChanged(after, result);
    return success();
  }

protected:
  void setToEntryState(TestLattice *lattice) override {}
};

class TestAnalysisPass
    : public impl::TestAnalysisPassBase<TestAnalysisPass> {
  raw_ostream &os;

public:
  explicit TestAnalysisPass(llvm::raw_ostream &ostream)
      : impl::TestAnalysisPassBase<TestAnalysisPass>(),
        os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    getOperation()->walk<WalkOrder::PreOrder>([&](FuncDefOp fnOp){
      Region &fnBody = fnOp.getFunctionBody();
      if (fnBody.empty()) {
        return WalkResult::skip();
      }

      DataFlowSolver solver;
      solver.load<mlir::dataflow::SparseConstantPropagation>();
      solver.load<mlir::dataflow::DeadCodeAnalysis>();
      solver.load<TestAnalysis>(os);
      LogicalResult res = solver.initializeAndRun(fnOp);
      if (res.failed()) {
        llvm::report_fatal_error("TestAnalysis failed.");
      }

      ProgramPoint *point = solver.getProgramPointAfter(fnBody.back().getTerminator());
      TestLattice *finalLattice = solver.getOrCreateState<TestLattice>(point);

      printOp(os, fnOp) << ":\n" << *finalLattice << '\n';

      return WalkResult::skip();
    });
  }
};

std::unique_ptr<mlir::Pass> createTestAnalysisPass() {
  return std::make_unique<TestAnalysisPass>(llvm::errs());
}

} // namespace llzk
