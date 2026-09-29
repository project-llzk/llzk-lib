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
#include "r1cs/Transforms/TransformationPassPipelines.h"
#include "r1cs/Transforms/TransformationPasses.h"
#include "smt/Conversions/ConversionPasses.h"
#include "tools/config.h"
#include "zklean/Conversions/Passes.h"
#include "zklean/DialectRegistration.h"

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Config/Config.h"
#include "llzk/Dialect/Array/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Bool/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.h"
#include "llzk/Dialect/Include/Util/IncludeHelper.h"
#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Polymorphic/Transforms/TransformationPasses.h"
#include "llzk/Dialect/Struct/Transforms/TransformationPasses.h"
#include "llzk/Transforms/LLZKTransformationPassPipelines.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Transforms/SpecializedMemoryPasses.h"
#include "llzk/Validators/LLZKValidationPasses.h"

#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <string>
#include <tuple>

#if LLZK_WITH_PCL
#include "pcl/Conversion/ConversionPasses.h"
#include "pcl/DialectRegistration.h"
#include "pcl/Transforms/TransformationPasses.h"
#endif // LLZK_WITH_PCL

static llvm::cl::list<std::string> IncludeDirs(
    "I", llvm::cl::desc("Directory of include files"), llvm::cl::value_desc("directory"),
    llvm::cl::Prefix
);

static llvm::cl::opt<bool>
    PrintAllOps("print-llzk-ops", llvm::cl::desc("Print a list of all ops registered in LLZK"));

/// Replace `mlir::registerTransformsPasses()` to register a custom `remove-dead-values` pass
/// because MLIR version 23.1.0 has a bug where the pass tracks `poison` values that it created
/// during its current invocation only and may end up leaving behind dead `poison` values.
namespace mlir_patch {

inline static void registerTransformsPasses() {
  mlir::registerBubbleDownMemorySpaceCastsPass();
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCompositeFixedPointPass();
  mlir::registerControlFlowSinkPass();
  mlir::registerGenerateRuntimeVerificationPass();
  mlir::registerInlinerPass();
  mlir::registerLocationSnapshotPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerLoopInvariantSubsetHoistingPass();
  mlir::registerMem2RegPass();
  mlir::registerPrintIRPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerPass(llzk::createRemoveDeadValuesWorkaroundPass);
  mlir::registerSCCPPass();
  mlir::registerSROAPass();
  mlir::registerStripDebugInfoPass();
  mlir::registerSymbolDCEPass();
  mlir::registerSymbolPrivatizePass();
  mlir::registerTopologicalSortPass();
  mlir::registerTrivialDeadCodeEliminationPass();
  mlir::registerViewOpGraphPass();
}

} // namespace mlir_patch

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
  // registers CSE, etc
  mlir_patch::registerTransformsPasses();
  llzk::registerAllDialects(registry);
  r1cs::registerAllDialects(registry);
  zklean::registerAllDialects(registry);
  mlir::func::registerInlinerExtension(registry);
#if LLZK_WITH_PCL
  pcl::registerAllDialects(registry);
#endif // LLZK_WITH_PCL

  llzk::registerValidationPasses();
  llzk::registerAnalysisPasses();
  llzk::registerTransformationPasses();
  llzk::array::registerTransformationPasses();
  llzk::component::registerTransformationPasses();
  llzk::boolean::registerTransformationPasses();
  llzk::global::registerTransformationPasses();
  llzk::include::registerTransformationPasses();
  llzk::polymorphic::registerTransformationPasses();
  llzk::pod::registerTransformationPasses();
  r1cs::registerTransformationPasses();
  zklean::registerConversionPasses();
#if LLZK_WITH_PCL
  pcl::registerPCLConversionPasses();
  pcl::registerTransformationPasses();
#endif // LLZK_WITH_PCL
  llzk::smt::registerConversionPasses();

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
