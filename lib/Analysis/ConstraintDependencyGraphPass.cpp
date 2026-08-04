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

namespace llzk {
#define GEN_PASS_DEF_CONSTRAINTDEPENDENCYGRAPHPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"
} // namespace llzk

using namespace mlir;

namespace {

class PassImpl : public llzk::impl::ConstraintDependencyGraphPrinterPassBase<PassImpl> {
  using Base = ConstraintDependencyGraphPrinterPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    markAllAnalysesPreserved();

    if (!llvm::isa<ModuleOp>(getOperation())) {
      const char *msg = "ConstraintDependencyGraphPrinterPass error: should be run on ModuleOp!";
      getOperation()->emitError(msg).report();
      llvm::report_fatal_error(msg);
    }

    auto &cs = getAnalysis<llzk::ConstraintDependencyGraphModuleAnalysis>();
    cs.setIntraprocedural(runIntraprocedural);
    auto am = getAnalysisManager();
    cs.ensureAnalysisRun(am);

    auto &os = llzk::toStream(outputStream);
    for (const auto &[s, cdg] : cs.getCurrentResults()) {
      auto &structDef = const_cast<llzk::component::StructDefOp &>(s);
      FailureOr<SymbolRefAttr> fullName = llzk::getPathFromTopRoot(structDef);
      llzk::ensure(
          succeeded(fullName),
          "could not resolve fully qualified name of struct " + Twine(structDef.getName())
      );
      os << fullName.value() << ' ';
      cdg.get().print(os);
    }
  }
};

} // namespace
