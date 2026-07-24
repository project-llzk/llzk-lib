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
#include "llzk/Util/SymbolHelper.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {
#define GEN_PASS_DEF_CALLGRAPHPRINTERPASS
#define GEN_PASS_DEF_CALLGRAPHSCCSPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"
} // namespace llzk

namespace {

class PassImpl : public llzk::impl::CallGraphPrinterPassBase<PassImpl> {
  using Base = CallGraphPrinterPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &cga = getAnalysis<llzk::CallGraphAnalysis>();
    cga.getCallGraph().print(llzk::toStream(outputStream));
  }
};

class SCCPassImpl : public llzk::impl::CallGraphSCCsPrinterPassBase<SCCPassImpl> {
  using Base = CallGraphSCCsPrinterPassBase<SCCPassImpl>;
  using Base::Base;

  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto &os = llzk::toStream(outputStream);
    auto &CG = getAnalysis<llzk::CallGraphAnalysis>();
    unsigned sccNum = 0;
    os << "SCCs for the program in PostOrder:";
    for (auto SCCI = llvm::scc_begin<const llzk::CallGraph *>(&CG.getCallGraph()); !SCCI.isAtEnd();
         ++SCCI) {
      const std::vector<const llzk::CallGraphNode *> &nextSCC = *SCCI;
      os << "\nSCC #" << ++sccNum << ": ";
      bool First = true;
      for (const llzk::CallGraphNode *CGN : nextSCC) {
        if (First) {
          First = false;
        } else {
          os << ", ";
        }
        if (CGN->isExternal()) {
          os << "external node";
        } else {
          mlir::CallableOpInterface calledFn = CGN->getCalledFunction();
          auto calledSym = llvm::dyn_cast<mlir::SymbolOpInterface>(calledFn.getOperation());
          assert(calledSym && "call graph nodes must refer to callable symbols");
          os << llzk::getFullyQualifiedName(calledSym);
        }
      }

      if (nextSCC.size() == 1 && SCCI.hasCycle()) {
        os << " (Has self-loop).";
      }
    }
    os << '\n';
  }
};

} // namespace
