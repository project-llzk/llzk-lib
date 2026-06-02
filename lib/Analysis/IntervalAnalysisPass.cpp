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
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/AsmState.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

namespace llzk {

#define DEBUG_TYPE "llzk-interval-analysis-pass"

#define GEN_PASS_DECL_INTERVALANALYSISPRINTERPASS
#define GEN_PASS_DEF_INTERVALANALYSISPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"

using namespace component;
using namespace function;

class IntervalAnalysisPrinterPass
    : public impl::IntervalAnalysisPrinterPassBase<IntervalAnalysisPrinterPass> {
  llvm::raw_ostream &os;

public:
  explicit IntervalAnalysisPrinterPass(llvm::raw_ostream &ostream)
      : impl::IntervalAnalysisPrinterPassBase<IntervalAnalysisPrinterPass>(), os(ostream) {}

protected:
  void runOnOperation() override {
    markAllAnalysesPreserved();

    // Suppress false positive from `clang-tidy`
    // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
    auto modOp = llvm::dyn_cast<mlir::ModuleOp>(getOperation());
    if (!modOp) {
      constexpr const char *msg = "IntervalAnalysisPrinterPass error: should be run on ModuleOp!";
      getOperation()->emitError(msg).report();
      return;
    }

    // Initialize to the fallback field value
    FieldRef selectedField = Field::getField("bn128");
    if (!fieldName.empty()) {
      auto fieldLookupRes = Field::tryGetField(fieldName.c_str());
      if (mlir::failed(fieldLookupRes)) {
        modOp->emitError()
            .append(
                "IntervalAnalysisPrinterPass error: unknown field \"", fieldName, "\" specified"
            )
            .report();
        return;
      }
      selectedField = fieldLookupRes.value();
      LLVM_DEBUG(
          llvm::dbgs() << "[IntervalAnalysisPrinterPass] using explicit -field override '"
                       << selectedField.get().name() << "'\n";
      );
    } else if (auto detectedField = tryDetectSpecifiedField(modOp)) {
      selectedField = detectedField.value();
      LLVM_DEBUG(
          llvm::dbgs() << "[IntervalAnalysisPrinterPass] detected module field '"
                       << selectedField.get().name() << "' from module felt usage\n";
      );
    } else {
      modOp->emitWarning() << "could not detect a unique module field; falling back to '"
                           << selectedField.get().name() << '\'';
      LLVM_DEBUG(
          llvm::dbgs() << "[IntervalAnalysisPrinterPass] no explicit or detectable module field; "
                          "falling back to '"
                       << selectedField.get().name() << "'\n";
      );
    }

    auto &mia = getAnalysis<ModuleIntervalAnalysis>();
    mia.setField(selectedField);
    mia.setPropagateInputConstraints(propagateInputConstraints);
    mia.setTrackUnreducedIntervals(printUnreducedIntervals);
    auto am = getAnalysisManager();
    mia.ensureAnalysisRun(am);
    mlir::AsmState asmState(modOp);

    auto printValueInterval = [this, &asmState,
                               &mia](mlir::raw_ostream &out, int indent, mlir::Value value) {
      if (llvm::isa<llzk::array::ArrayType, StructType>(value.getType())) {
        return;
      }
      const auto *lattice = mia.getSolver().lookupState<IntervalAnalysisLattice>(value);
      if (!lattice) {
        return;
      }
      const ExpressionValue &expr = lattice->getValue().getScalarValue();
      out << '\n';
      out.indent(indent);
      value.printAsOperand(out, asmState);
      if (auto opResult = llvm::dyn_cast<mlir::OpResult>(value)) {
        out << " [" << opResult.getOwner()->getName().getStringRef() << "]";
      }
      out << " in " << expr.getInterval();
      if (printUnreducedIntervals && expr.hasUnreducedInterval()) {
        out << " ( unreduced: " << expr.getUnreducedInterval() << " )";
      }
    };

    auto printFunctionSSAIntervals =
        [&printValueInterval](mlir::raw_ostream &out, FuncDefOp fn, llvm::StringRef fnName) {
      if (!fn) {
        return;
      }

      out << '\n';
      out.indent(4) << fnName << " {";
      for (mlir::BlockArgument arg : fn.getArguments()) {
        printValueInterval(out, 8, arg);
      }
      fn.walk([&](mlir::Operation *op) {
        if (op == fn.getOperation()) {
          return;
        }
        for (mlir::Value result : op->getResults()) {
          printValueInterval(out, 8, result);
        }
      });
      out << '\n';
      out.indent(4) << '}';
    };

    for (const auto &[s, si] : mia.getCurrentResults()) {
      auto &structDef = const_cast<StructDefOp &>(s);
      auto fullName = getPathFromTopRoot(structDef);
      ensure(
          mlir::succeeded(fullName),
          "could not resolve fully qualified name of struct " + mlir::Twine(structDef.getName())
      );
      os << fullName.value() << ' ';
      si.get().print(os, printSolverConstraints, printComputeIntervals, printUnreducedIntervals);
      if (printSSAIntervals) {
        os << fullName.value() << " SSAIntervals {";
        if (printComputeIntervals) {
          printFunctionSSAIntervals(os, structDef.getComputeFuncOp(), FUNC_NAME_COMPUTE);
        }
        printFunctionSSAIntervals(os, structDef.getConstrainFuncOp(), FUNC_NAME_CONSTRAIN);
        if (auto productFn = structDef.getProductFuncOp();
            productFn && (!structDef.getConstrainFuncOp() || printComputeIntervals)) {
          printFunctionSSAIntervals(os, productFn, FUNC_NAME_PRODUCT);
        }
        os << "\n}\n";
      }
    }
  }
};

std::unique_ptr<mlir::Pass>
createIntervalAnalysisPrinterPass(llvm::raw_ostream &os = llvm::errs()) {
  return std::make_unique<IntervalAnalysisPrinterPass>(os);
}

} // namespace llzk
