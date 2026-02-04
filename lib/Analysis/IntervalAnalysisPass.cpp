//===-- IntervalAnalysisPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-interval-analysis` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolHelper.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

#define GEN_PASS_DECL_INTERVALANALYSISPRINTERPASS
#define GEN_PASS_DEF_INTERVALANALYSISPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

using namespace component;

class IntervalAnalysisPrinterPass
    : public impl::IntervalAnalysisPrinterPassBase<IntervalAnalysisPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit IntervalAnalysisPrinterPass(llvm::raw_ostream &ostream)
      : impl::IntervalAnalysisPrinterPassBase<IntervalAnalysisPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    auto modOp = llvm::dyn_cast<mlir::ModuleOp>(getOperation());
    if (!modOp) {
      constexpr const char *msg = "IntervalAnalysisPrinterPass error: should be run on ModuleOp!";
      getOperation()->emitError(msg).report();
      return;
    }

    const Field &field = Field::getField(fieldName.c_str());

    auto supportedFieldsRes = getSupportedFields(modOp);
    if (mlir::failed(supportedFieldsRes)) {
      std::string msg = (llvm::Twine("IntervalAnalysisPrinterPass error: could not parse \"") +
                         FIELD_ATTR_NAME + "\" attribute")
                            .str();
      modOp->emitError(msg).report();
      return;
    }

    const auto &supportedFields = supportedFieldsRes.value();
    if (!supportsField(supportedFields, field)) {
      std::string msg;
      llvm::raw_string_ostream ss(msg);
      ss << "IntervalAnalysisPrinterPass warning: circuit does not support field \"" << fieldName
         << "\", so analysis results may be inaccurate. Supported fields: [ ";
      llvm::interleaveComma(supportedFields, ss, [&ss](auto f) { ss << f.get().name(); });
      ss << " ]";
      modOp->emitWarning(msg).report();
    }

    auto &mia = getAnalysis<ModuleIntervalAnalysis>();
    mia.setField(field);
    mia.setPropagateInputConstraints(propagateInputConstraints);
    auto am = getAnalysisManager();
    mia.ensureAnalysisRun(am);

    for (auto &[s, si] : mia.getCurrentResults()) {
      auto &structDef = const_cast<StructDefOp &>(s);
      auto fullName = getPathFromTopRoot(structDef);
      ensure(
          mlir::succeeded(fullName),
          "could not resolve fully qualified name of struct " + mlir::Twine(structDef.getName())
      );
      os << fullName.value() << ' ';
      si.get().print(os, printSolverConstraints, printComputeIntervals);
    }
  }
};

std::unique_ptr<mlir::Pass>
createIntervalAnalysisPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<IntervalAnalysisPrinterPass>(os);
}

} // namespace llzk
