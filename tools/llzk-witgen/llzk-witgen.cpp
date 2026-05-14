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
#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/SMT/IR/SMTDialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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
  MLIRContext context;
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
  context.getOrLoadDialect<mlir::BuiltinDialect>();
  context.getOrLoadDialect<llzk::LLZKDialect>();
  context.getOrLoadDialect<llzk::array::ArrayDialect>();
  context.getOrLoadDialect<llzk::boolean::BoolDialect>();
  context.getOrLoadDialect<llzk::cast::CastDialect>();
  context.getOrLoadDialect<llzk::component::StructDialect>();
  context.getOrLoadDialect<llzk::constrain::ConstrainDialect>();
  context.getOrLoadDialect<llzk::felt::FeltDialect>();
  context.getOrLoadDialect<llzk::function::FunctionDialect>();
  context.getOrLoadDialect<llzk::global::GlobalDialect>();
  context.getOrLoadDialect<llzk::ram::RAMDialect>();
  context.getOrLoadDialect<llzk::include::IncludeDialect>();
  context.getOrLoadDialect<llzk::string::StringDialect>();
  context.getOrLoadDialect<llzk::pod::PODDialect>();
  context.getOrLoadDialect<llzk::polymorphic::PolymorphicDialect>();
  context.getOrLoadDialect<llzk::smt::SMTDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
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
