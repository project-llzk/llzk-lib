//===-- llzk-translate.cpp - LLZK translate tool ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a version of the mlir-translate tool configured for 
/// use on LLZK files.
///
//===----------------------------------------------------------------------===//

// #include "smt/Target/TranslateRegistration.h"
#include "zklean/Target/TranslateRegistration.h"
#include "tools/config.h"
#include "llzk/Config/Config.h"

#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-translate/MlirTranslateMain.h>
#include <mlir/Transforms/Passes.h>
#include <mlir/InitAllTranslations.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>

#if LLZK_WITH_PCL
// TODO
#endif // LLZK_WITH_PCL

// using namespace llzk;

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

//   // MLIR initialization
//   mlir::DialectRegistry registry;
//   // registers CSE, etc
//   mlir_hotfix::registerTransformsPasses();
//   llzk::registerAllDialects(registry);
//   r1cs::registerAllDialects(registry);
//   zklean::registerAllDialects(registry);
//   mlir::func::registerInlinerExtension(registry);
// #if LLZK_WITH_PCL
//   pcl::registerAllDialects(registry);
// #endif // LLZK_WITH_PCL
//
//   llzk::registerValidationPasses();
//   llzk::registerAnalysisPasses();
//   llzk::registerTransformationPasses();
//   llzk::component::registerTransformationPasses();
//   llzk::array::registerTransformationPasses();
//   llzk::include::registerTransformationPasses();
//   llzk::polymorphic::registerTransformationPasses();
//   llzk::pod::registerTransformationPasses();
//   r1cs::registerTransformationPasses();
//   zklean::registerConversionPasses();
//   zklean::registerZKLeanPasses();
// #if LLZK_WITH_PCL
//   pcl::registerTransformationPasses();
//   pcl::conversion::registerPCLTransformationPasses();
// #endif // LLZK_WITH_PCL
//   llzk::smt::registerConversionPasses();
//
//   llzk::registerTransformationPassPipelines();
//   r1cs::registerTransformationPassPipelines();
//   llzk::smt::registerSMTPasses();

  // Register all MLIR translations 
  mlir::registerAllTranslations();
  // smt::registerSmtTranslation();
  zklean::registerZKLeanTranslation();

  // Run 'mlir-translate'
  return failed(mlir::mlirTranslateMain(argc, argv, "LLZK Translation tool"));
}
