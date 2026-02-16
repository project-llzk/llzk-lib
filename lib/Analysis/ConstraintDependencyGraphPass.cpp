//===-- ConstraintDependencyGraphPass.cpp -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-constraint-dependency-graphs` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Util/SymbolHelper.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/ErrorHandling.h>

using namespace mlir;

namespace llzk {

using namespace component;

#define GEN_PASS_DECL_CONSTRAINTDEPENDENCYGRAPHPRINTERPASS
#define GEN_PASS_DEF_CONSTRAINTDEPENDENCYGRAPHPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

class ConstraintDependencyGraphPrinterPass
    : public impl::ConstraintDependencyGraphPrinterPassBase<ConstraintDependencyGraphPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit ConstraintDependencyGraphPrinterPass(llvm::raw_ostream &ostream)
      : impl::ConstraintDependencyGraphPrinterPassBase<ConstraintDependencyGraphPrinterPass>(),
        os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    if (!llvm::isa<mlir::ModuleOp>(getOperation())) {
      const char *msg = "ConstraintDependencyGraphPrinterPass error: should be run on ModuleOp!";
      getOperation()->emitError(msg).report();
      llvm::report_fatal_error(msg);
    }

    auto &cs = getAnalysis<ConstraintDependencyGraphModuleAnalysis>();
    cs.setIntraprocedural(runIntraprocedural);
    auto am = getAnalysisManager();
    cs.ensureAnalysisRun(am);
    for (auto &[s, cdg] : cs.getCurrentResults()) {
      auto &structDef = const_cast<StructDefOp &>(s);
      FailureOr<SymbolRefAttr> fullName = getPathFromTopRoot(structDef);
      ensure(
          mlir::succeeded(fullName),
          "could not resolve fully qualified name of struct " + mlir::Twine(structDef.getName())
      );
      os << fullName.value() << ' ';
      cdg.get().print(os);
    }
  }
};

std::unique_ptr<mlir::Pass>
createConstraintDependencyGraphPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<ConstraintDependencyGraphPrinterPass>(os);
}

} // namespace llzk
