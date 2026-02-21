//===-- llzk-opt.cpp - LLZK opt tool ----------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a version of the mlir-opt tool configured for use on
/// LLZK files.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Config/Config.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"
#include "llzk/Dialect/Include/Util/IncludeHelper.h"
#include "llzk/Dialect/InitDialects.h"
#include "zklean/Conversions/Passes.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "zklean/Transforms/ZKLeanPasses.h"
#include "llzk/Validators/LLZKValidationPasses.h"
#include "r1cs/Dialect/IR/Dialect.h"
#include "r1cs/DialectRegistration.h"
#include "r1cs/Transforms/TransformationPasses.h"
#include "zklean/DialectRegistration.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>

#include "tools/config.h"

#if LLZK_WITH_PCL
#include <pcl/Dialect/IR/Dialect.h>
#include <pcl/InitAllDialects.h>
#include <pcl/Transforms/PCLTransformationPasses.h>
#endif // LLZK_WITH_PCL

static llvm::cl::list<std::string> IncludeDirs(
    "I", llvm::cl::desc("Directory of include files"), llvm::cl::value_desc("directory"),
    llvm::cl::Prefix
);

static llvm::cl::opt<bool>
    PrintAllOps("print-llzk-ops", llvm::cl::desc("Print a list of all ops registered in LLZK"));

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

  // MLIR initialization
  mlir::DialectRegistry registry;
  llzk::registerAllDialects(registry);
  r1cs::registerAllDialects(registry);
  zklean::registerAllDialects(registry);
#if LLZK_WITH_PCL
  pcl::registerAllDialects(registry);
#endif // LLZK_WITH_PCL

  llzk::registerValidationPasses();
  llzk::registerAnalysisPasses();
  llzk::registerTransformationPasses();
  llzk::registerLLZKConversionPasses();
  llzk::array::registerTransformationPasses();
  llzk::include::registerTransformationPasses();
  llzk::polymorphic::registerTransformationPasses();
  r1cs::registerTransformationPasses();
  llzk::zklean::registerZKLeanPasses();
#if LLZK_WITH_PCL
  pcl::registerTransformationPasses();
#endif // LLZK_WITH_PCL

  llzk::registerTransformationPassPipelines();
  r1cs::registerTransformationPassPipelines();

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "llzk-opt", registry);

  if (PrintAllOps) {
    mlir::MLIRContext context;
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
    llvm::outs() << "All ops registered in LLZK IR: {\n";
    for (const auto &opName : context.getRegisteredOperations()) {
      llvm::outs().indent(2) << opName.getStringRef() << '\n';
    }
    llvm::outs() << "}\n";
    return EXIT_SUCCESS;
  }

  // Set the include directories from CL option
  if (mlir::failed(llzk::GlobalSourceMgr::get().setup(IncludeDirs))) {
    return EXIT_FAILURE;
  }

  // Run 'mlir-opt'
  auto result = mlir::MlirOptMain(argc, argv, inputFilename, outputFilename, registry);
  return mlir::asMainReturnCode(result);
}
