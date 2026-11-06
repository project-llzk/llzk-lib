//===-- llzk-translate.cpp - LLZK translate tool ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a version of the mlir-translate tool configured for
/// use on LLZK files.
///
//===----------------------------------------------------------------------===//

#include "r1cs/Target/TranslateRegistration.h"
#include "smt/Target/TranslateRegistration.h"
#include "tools/config.h"
#include "zklean/Target/TranslateRegistration.h"

#include "llzk/Config/Config.h"
#include "llzk/Dialect/DialectRegistration.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Util/LLZKLayout.h"

#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/InitAllTranslations.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Tools/mlir-translate/Translation.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>

#if LLZK_WITH_PCL
#include "pcl/Target/TranslateRegistration.h"
#endif // LLZK_WITH_PCL

using namespace llzk;
using namespace mlir;

/// Optionally monomorphize the module before exporting its logical signal layout.
static LogicalResult generateLayout(Operation *op, llvm::raw_ostream &output, bool monomorphize) {
  auto module = dyn_cast<ModuleOp>(op);
  if (!module) {
    return op->emitOpError("expected builtin.module as top level operation");
  }
  if (monomorphize) {
    PassManager pm(module.getContext());
    pm.addPass(polymorphic::createTemplateMonomorphizationPass());
    if (failed(pm.run(module))) {
      return failure();
    }
  }
  auto result = buildLLZKLayout(module);
  if (failed(result)) {
    return failure();
  }
  printLLZKLayout(*result, output);
  return success();
}

int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace, relevant LLZK files,"
      " and associated run script(s).\n"
  );
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << "\nLLZK (" LLZK_URL "):\n  LLZK version " LLZK_VERSION_STRING "\n";
  });

  // Register all MLIR translations
  registerAllTranslations();
  TranslateFromMLIRRegistration layout(
      "llzk-layout", "export the logical LLZK signal layout",
      [](Operation *op, llvm::raw_ostream &output) { return generateLayout(op, output, false); },
      [](DialectRegistry &registry) { registerDialects(registry); }
  );
  TranslateFromMLIRRegistration generateLayoutPipeline(
      "llzk-generate-layout", "monomorphize LLZK and export the logical signal layout",
      [](Operation *op, llvm::raw_ostream &output) { return generateLayout(op, output, true); },
      [](DialectRegistry &registry) { registerDialects(registry); }
  );
  r1cs::registerR1CSTranslation();
  llzk::smt::registerSmtTranslation();
  zklean::registerZKLeanTranslation();
#if LLZK_WITH_PCL
  pcl::registerPclTranslation();
#endif

  // Run 'mlir-translate'
  return failed(mlirTranslateMain(argc, argv, "LLZK Translation tool"));
}
