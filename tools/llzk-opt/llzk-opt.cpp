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

#include "r1cs/DialectRegistration.h"
#include "tools/config.h"
#include "zklean/DialectRegistration.h"

#include "llzk/Config/Config.h"
#include "llzk/Dialect/DialectRegistration.h"
#include "llzk/Dialect/Include/Util/IncludeHelper.h"

#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <exception>
#include <string>
#include <tuple>

#if LLZK_WITH_PCL
#include "pcl/DialectRegistration.h"
#endif // LLZK_WITH_PCL

/// Register options and passes, then run the LLZK optimizer.
static int runMain(int argc, char **argv) {
  llvm::cl::list<std::string> IncludeDirs(
      "I", llvm::cl::desc("Directory of include files"), llvm::cl::value_desc("directory"),
      llvm::cl::Prefix
  );

  llvm::cl::opt<bool> PrintAllOps(
      "print-llzk-ops", llvm::cl::desc("Print a list of all ops registered in LLZK")
  );

  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace, relevant LLZK files,"
      " and associated run script(s).\n"
  );
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << "\nLLZK (" LLZK_URL "):\n  LLZK version " LLZK_VERSION_STRING "\n";
  });

  // Register dialects and passes
  mlir::DialectRegistry registry;
  llzk::registerDialects(registry);
  r1cs::registerDialects(registry);
  zklean::registerDialects(registry);
#if LLZK_WITH_PCL
  pcl::registerDialects(registry);
#endif // LLZK_WITH_PCL

  llzk::registerPasses(registry);
  r1cs::registerPasses(registry);
  zklean::registerPasses(registry);
#if LLZK_WITH_PCL
  pcl::registerPasses(registry);
#endif // LLZK_WITH_PCL

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

int main(int argc, char **argv) noexcept {
  try {
    return runMain(argc, argv);
  } catch (const std::exception &ex) {
    llvm::errs() << "llzk-opt: unhandled exception: " << ex.what() << '\n';
  } catch (...) {
    llvm::errs() << "llzk-opt: unhandled non-standard exception\n";
  }
  return EXIT_FAILURE;
}
