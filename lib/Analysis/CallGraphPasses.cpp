//===-- CallGraphPasses.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The contents of this file are adapted from llvm/lib/Analysis/CallGraph.cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-call-graph` and
/// `-llzk-print-call-graph-sccs` passes.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/CallGraphAnalyses.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

#define GEN_PASS_DEF_CALLGRAPHPRINTERPASS
#define GEN_PASS_DEF_CALLGRAPHSCCSPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

class CallGraphPrinterPass : public impl::CallGraphPrinterPassBase<CallGraphPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit CallGraphPrinterPass(llvm::raw_ostream &ostream)
      : impl::CallGraphPrinterPassBase<CallGraphPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &cga = getAnalysis<CallGraphAnalysis>();
    cga.getCallGraph().print(os);
  }
};

std::unique_ptr<mlir::Pass> createCallGraphPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<CallGraphPrinterPass>(os);
}

class CallGraphSCCsPrinterPass
    : public impl::CallGraphSCCsPrinterPassBase<CallGraphSCCsPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit CallGraphSCCsPrinterPass(llvm::raw_ostream &ostream)
      : impl::CallGraphSCCsPrinterPassBase<CallGraphSCCsPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &CG = getAnalysis<CallGraphAnalysis>();
    unsigned sccNum = 0;
    os << "SCCs for the program in PostOrder:";
    for (auto SCCI = llvm::scc_begin<const llzk::CallGraph *>(&CG.getCallGraph()); !SCCI.isAtEnd();
         ++SCCI) {
      const std::vector<const CallGraphNode *> &nextSCC = *SCCI;
      os << "\nSCC #" << ++sccNum << ": ";
      bool First = true;
      for (const CallGraphNode *CGN : nextSCC) {
        if (First) {
          First = false;
        } else {
          os << ", ";
        }
        if (CGN->isExternal()) {
          os << "external node";
        } else {
          os << CGN->getCalledFunction().getFullyQualifiedName();
        }
      }

      if (nextSCC.size() == 1 && SCCI.hasCycle()) {
        os << " (Has self-loop).";
      }
    }
    os << '\n';
  }
};

std::unique_ptr<mlir::Pass> createCallGraphSCCsPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<CallGraphSCCsPrinterPass>(os);
}

} // namespace llzk
