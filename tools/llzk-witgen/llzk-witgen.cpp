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

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/Include/Util/IncludeHelper.h"
#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/StringExtras.h>
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
static llvm::cl::opt<std::string> BackendName(
    "backend", llvm::cl::desc("Execution backend: interpreter or execution-engine"),
    llvm::cl::init("interpreter")
);
static llvm::cl::opt<std::string> OutputScopeName(
    "output-scope", llvm::cl::desc("Output scope: public or full-witness"), llvm::cl::init("public")
);
static llvm::cl::opt<std::string> UninitializedBehaviorName(
    "uninitialized-behavior", llvm::cl::desc("Uninitialized value behavior: zero, random, or fail"),
    llvm::cl::init("zero")
);
static llvm::cl::opt<uint64_t>
    UninitializedSeed("uninitialized-seed", llvm::cl::desc("Seed for random uninitialized values"));
static llvm::cl::opt<bool>
    DumpJITCore("dump-jit-core", llvm::cl::desc("Print the pre-LLVM JIT module"));
static llvm::cl::opt<bool>
    DumpJITLLVM("dump-jit-llvm", llvm::cl::desc("Print the post-LLVM JIT module"));

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
  mlir::func::registerInlinerExtension(registry);
  registry.insert<
      mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  context.loadDialect<
      mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect, mlir::func::FuncDialect,
      mlir::memref::MemRefDialect, mlir::scf::SCFDialect>();
  if (failed(llzk::GlobalSourceMgr::get().setup(IncludeDirs))) {
    return EXIT_FAILURE;
  }

  auto sourceBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InputFilename);
  if (!sourceBuffer) {
    llvm::errs() << sourceBuffer.getError().message() << '\n';
    return EXIT_FAILURE;
  }

  ParserConfig parserConfig(&context);
  OwningOpRef<ModuleOp> moduleOp =
      parseSourceString<ModuleOp>(sourceBuffer.get()->getBuffer(), parserConfig, InputFilename);
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

  llzk::witgen::WitgenOptions options;
  if (BackendName == "execution-engine") {
    options.backend = llzk::witgen::Backend::ExecutionEngine;
  } else if (BackendName == "interpreter") {
    options.backend = llzk::witgen::Backend::Interpreter;
  } else {
    llvm::errs() << "unknown backend: " << BackendName << '\n';
    return EXIT_FAILURE;
  }
  if (OutputScopeName == "full-witness") {
    options.outputScope = llzk::witgen::OutputScope::FullWitness;
  } else if (OutputScopeName == "public") {
    options.outputScope = llzk::witgen::OutputScope::Public;
  } else {
    llvm::errs() << "unknown output scope: " << OutputScopeName << '\n';
    return EXIT_FAILURE;
  }
  if (UninitializedBehaviorName == "zero") {
    options.uninitializedBehavior = llzk::witgen::UninitializedBehavior::Zero;
  } else if (UninitializedBehaviorName == "random") {
    options.uninitializedBehavior = llzk::witgen::UninitializedBehavior::Random;
  } else if (UninitializedBehaviorName == "fail") {
    options.uninitializedBehavior = llzk::witgen::UninitializedBehavior::Fail;
  } else {
    llvm::errs() << "unknown uninitialized behavior: " << UninitializedBehaviorName << '\n';
    return EXIT_FAILURE;
  }
  if (UninitializedSeed.getNumOccurrences() > 0) {
    options.randomSeed = UninitializedSeed;
  }
  options.inlineIncludes = true;
  options.dumpJITCore = DumpJITCore;
  options.dumpJITLLVM = DumpJITLLVM;

  auto result = llzk::witgen::runWitgen(*moduleOp, *parsed, options);
  if (!result) {
    llvm::errs() << "llzk-witgen error: " << llvm::toString(result.takeError()) << '\n';
    return EXIT_FAILURE;
  }

  llvm::outs() << llvm::formatv("{0:2}", *result) << '\n';
  return EXIT_SUCCESS;
}
