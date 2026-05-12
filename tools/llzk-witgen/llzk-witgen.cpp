//===-- llzk-witgen.cpp - LLZK witness generation tool ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "WitgenDriver.h"
#include "tools/config.h"

#include "llzk/Dialect/Include/Util/IncludeHelper.h"
#include "llzk/Dialect/InitDialects.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FormatVariadic.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/PrettyStackTrace.h>
#include <llvm/Support/Signals.h>

using namespace mlir;

static llvm::cl::opt<std::string> InputFilename(llvm::cl::Positional, llvm::cl::Required);
static llvm::cl::opt<std::string>
    InputsFilename("inputs", llvm::cl::Required, llvm::cl::desc("JSON input file"));
static llvm::cl::list<std::string> IncludeDirs(
    "I", llvm::cl::desc("Directory of include files"), llvm::cl::value_desc("directory"),
    llvm::cl::Prefix
);

/// Execute the llzk-witgen command-line tool.
int main(int argc, char **argv) {
  llvm::sys::PrintStackTraceOnErrorSignal(llvm::StringRef());
  llvm::setBugReportMsg(
      "PLEASE submit a bug report to " BUG_REPORT_URL
      " and include the crash backtrace, relevant LLZK files, and associated run script(s).\n"
  );

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "llzk-witgen: execute LLZK compute semantics and emit JSON public outputs.\n"
      "Note: llzk-witgen v1 ignores constrain() and traps on bool.assert.\n"
  );

  DialectRegistry registry;
  llzk::registerAllDialects(registry);
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  if (failed(llzk::GlobalSourceMgr::get().setup(IncludeDirs))) {
    return EXIT_FAILURE;
  }

  OwningOpRef<ModuleOp> moduleOp = parseSourceFile<ModuleOp>(InputFilename, &context);
  if (!moduleOp) {
    return EXIT_FAILURE;
  }

  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(InputsFilename);
  if (!buffer) {
    llvm::errs() << buffer.getError().message() << '\n';
    return EXIT_FAILURE;
  }

  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed) {
    llvm::errs() << "failed to parse JSON input: " << llvm::toString(parsed.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  auto result = llzk::witgen::runWitgen(*moduleOp, *parsed, /*inlineIncludes=*/true);
  if (!result) {
    llvm::errs() << "llzk-witgen error: " << llvm::toString(result.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  llvm::outs() << llvm::formatv("{0:2}", *result) << '\n';
  return EXIT_SUCCESS;
}
