//===-- Parsers.h -----------------------------------------------*- C++ -*-===//
//
// Command line parsers for LLZK transformation passes.
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Compare.h"

#include <mlir/Pass/Pass.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

#include <string>

namespace llzk {

namespace detail {

/// Shared storage and helpers for nested textual pass and pipeline options.
struct NestedTextualOptions {
  /// The validated textual form without the outer `{...}` delimiters.
  std::string str;

  /// Recreate a nested value from the stored option string and re-validate it.
  ///
  /// The CLI parser validates `str` before storing it, but callers still need a
  /// fresh initialized pass or pipeline options object when materializing the
  /// nested configuration later.
  template <typename ValueT, typename CreateFnT, typename InitializeFnT>
  std::unique_ptr<ValueT> createValidatedValue(
      CreateFnT &&createValue, llvm::StringRef kind, InitializeFnT &&initializeValue
  ) const {
    auto value = createValue();
    if (str.empty()) {
      return value;
    }

    std::string error;
    if (failed(initializeValue(*value, str, error))) {
      llvm::report_fatal_error(
          llvm::Twine("failed to initialize previously-validated nested ") + kind +
          " options: " + error
      );
    }
    return value;
  }
};

} // namespace detail

/// Stores textual options for a constituent pass after validating them against
/// that pass' native MLIR option parser.
template <auto CreatePass> struct NestedPassOptions : detail::NestedTextualOptions {
  /// Build a fresh pass instance with the validated options applied.
  std::unique_ptr<mlir::Pass> createPass() const {
    return this->createValidatedValue<mlir::Pass>(CreatePass, "pass", initializePass);
  }

  static mlir::LogicalResult
  initializePass(mlir::Pass &pass, llvm::StringRef options, std::string &error) {
    return pass.initializeOptions(options, [&error](const llvm::Twine &message) {
      error = message.str();
      return mlir::failure();
    });
  }
};

/// Stores textual options for a constituent pipeline after validating them
/// against that pipeline's native MLIR option parser.
template <typename PipelineOptionsT> struct NestedPipelineOptions : detail::NestedTextualOptions {
  /// Build a fresh options object with the validated options applied.
  std::unique_ptr<PipelineOptionsT> createOptions() const {
    return this->createValidatedValue<PipelineOptionsT>(
        std::make_unique<PipelineOptionsT>, "pipeline", initializeOptions
    );
  }

  static mlir::LogicalResult
  initializeOptions(PipelineOptionsT &options, llvm::StringRef value, std::string &error) {
    llvm::raw_string_ostream errorStream(error);
    return options.parseFromString(value, errorStream);
  }
};

} // namespace llzk

// Custom command line parsers
namespace llvm {
namespace cl {

template <typename OptionsT> class NestedOptionsParserBase : public basic_parser<OptionsT> {
public:
  NestedOptionsParserBase(Option &O) : basic_parser<OptionsT>(O) {}

protected:
  /// Parse a nested pass or pipeline option payload, optionally stripping a
  /// surrounding `{...}` wrapper.
  bool parseNestedOptions(Option &O, StringRef Arg, StringRef kind, StringRef &options) const {
    options = Arg;
    if (options.consume_front("{") && !options.consume_back("}")) {
      return O.error(llvm::Twine("expected nested ") + kind + " options to end with '}'");
    }
    return false;
  }

public:
  static void print(llvm::raw_ostream &OS, const OptionsT &Val) { OS << '{' << Val.str << '}'; }

  void printOptionDiff(
      const Option &O, const OptionsT &V, const OptionValue<OptionsT> &Default, size_t GlobalWidth
  ) const {
    this->printOptionName(O, GlobalWidth);
    print(llvm::outs(), V);
    llvm::outs() << " (default: ";
    if (Default.hasValue()) {
      print(llvm::outs(), Default.getValue());
    } else {
      llvm::outs() << "<unspecified>";
    }
    llvm::outs() << ")\n";
  }
};

/// Parser for textual options that are validated by a constituent MLIR pass.
template <auto CreatePass>
class parser<llzk::NestedPassOptions<CreatePass>>
    : public NestedOptionsParserBase<llzk::NestedPassOptions<CreatePass>> {
public:
  using OptionsT = llzk::NestedPassOptions<CreatePass>;
  using Base = NestedOptionsParserBase<OptionsT>;

  parser(Option &O) : Base(O) {}

  bool parse(Option &O, StringRef, StringRef Arg, OptionsT &Val) {
    StringRef options;
    if (this->parseNestedOptions(O, Arg, "pass", options)) {
      return true;
    }

    auto pass = CreatePass();
    std::string error;
    if (failed(OptionsT::initializePass(*pass, options, error))) {
      return O.error(error);
    }

    Val.str = options.str();
    return false;
  }
};

/// Parser for textual options that are validated by a constituent MLIR
/// pipeline.
template <typename PipelineOptionsT>
class parser<llzk::NestedPipelineOptions<PipelineOptionsT>>
    : public NestedOptionsParserBase<llzk::NestedPipelineOptions<PipelineOptionsT>> {
public:
  using OptionsT = llzk::NestedPipelineOptions<PipelineOptionsT>;
  using Base = NestedOptionsParserBase<OptionsT>;

  parser(Option &O) : Base(O) {}

  bool parse(Option &O, StringRef, StringRef Arg, OptionsT &Val) {
    StringRef options;
    if (this->parseNestedOptions(O, Arg, "pipeline", options)) {
      return true;
    }

    PipelineOptionsT pipelineOptions;
    std::string error;
    if (failed(OptionsT::initializeOptions(pipelineOptions, options, error))) {
      return O.error(error);
    }

    Val.str = options.str();
    return false;
  }
};

// Parser for APInt
template <> class parser<APInt> : public basic_parser<APInt> {
public:
  parser(Option &O) : basic_parser(O) {}

  bool parse(Option &O, StringRef, StringRef Arg, APInt &Val) {
    if (Arg.empty()) {
      return O.error("empty integer literal");
    }
    if (!all_of(Arg, [](char c) { return isDigit(c); })) {
      return O.error("arg must be in base 10 (digits).");
    }
    // Decimal-only: allocate a safe width then shrink.
    unsigned bits = std::max(1u, 4u * llzk::checkedCast<unsigned>(Arg.size()));
    APInt tmp(bits, Arg, 10);
    unsigned active = tmp.getActiveBits();
    if (active == 0) {
      active = 1;
    }
    Val = tmp.zextOrTrunc(active);
    return false;
  }

  // Prints how the passed option differs from the default one specified in the pass
  // For example, if V = 17 and Default = 11 then it should print
  // [OptionName] 17 (default: 11)
  void printOptionDiff(
      const Option &O, const APInt &V, const OptionValue<APInt> &Default, size_t GlobalWidth
  ) const {
    std::string Cur = llvm::toString(V, 10, false);

    std::string Def = "<unspecified>";
    if (Default.hasValue()) {
      const APInt &D = Default.getValue();
      Def = llvm::toString(D, 10, false);
    }

    printOptionName(O, GlobalWidth);
    llvm::outs() << Cur << " (default: " << Def << ")\n";
  }
};

} // namespace cl
} // namespace llvm
