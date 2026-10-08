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

/// Optionally monomorphize the module before exporting its logical signal layout.
static mlir::LogicalResult
generateLayout(mlir::Operation *op, llvm::raw_ostream &output, bool monomorphize) {
  auto module = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!module) {
    return op->emitOpError("expected builtin.module as top level operation");
  }
  if (monomorphize) {
    mlir::PassManager pm(module.getContext());
    pm.addPass(llzk::polymorphic::createTemplateMonomorphizationPass());
    if (mlir::failed(pm.run(module))) {
      return mlir::failure();
    }
  }
  auto result = llzk::buildLLZKLayout(module);
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  llzk::printLLZKLayout(*result, output);
  return mlir::success();
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
  mlir::registerAllTranslations();
  mlir::TranslateFromMLIRRegistration layout(
      "llzk-layout", "export the logical LLZK signal layout",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
    return generateLayout(op, output, false);
  }, [](mlir::DialectRegistry &registry) { llzk::registerDialects(registry); }
  );
  mlir::TranslateFromMLIRRegistration generateLayoutPipeline(
      "llzk-generate-layout", "monomorphize LLZK and export the logical signal layout",
      [](mlir::Operation *op, llvm::raw_ostream &output) {
    return generateLayout(op, output, true);
  }, [](mlir::DialectRegistry &registry) { llzk::registerDialects(registry); }
  );
  r1cs::registerR1CSTranslation();
  smt::registerSmtTranslation();
  zklean::registerZKLeanTranslation();
#if LLZK_WITH_PCL
  pcl::registerPclTranslation();
#endif

  // Run 'mlir-translate'
  return failed(mlir::mlirTranslateMain(argc, argv, "LLZK Translation tool"));
}
