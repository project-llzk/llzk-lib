//===-- Parsers.h -----------------------------------------------*- C++ -*-===//
//
// Command line parsers for LLZK transformation passes.
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
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

#include <string>

namespace llzk {

/// Stores textual options for a constituent pass after validating them against
/// that pass' native MLIR option parser.
template <auto CreatePass> struct NestedPassOptions {
  /// The validated textual form without the outer `{...}` delimiters.
  std::string value;

  /// Build a fresh pass instance with the validated options applied.
  std::unique_ptr<mlir::Pass> createPass() const {
    auto pass = CreatePass();
    if (value.empty()) {
      return pass;
    }

    std::string error;
    if (failed(initializePass(*pass, value, error))) {
      llvm::report_fatal_error(
          llvm::Twine("failed to initialize previously-validated nested pass options: ") + error
      );
    }
    return pass;
  }

  static mlir::LogicalResult
  initializePass(mlir::Pass &pass, llvm::StringRef options, std::string &error) {
    return pass.initializeOptions(options, [&error](const llvm::Twine &message) {
      error = message.str();
      return mlir::failure();
    });
  }
};

} // namespace llzk

// Custom command line parsers
namespace llvm {
namespace cl {

/// Parser for textual options that are validated by a constituent MLIR pass.
template <auto CreatePass>
class parser<llzk::NestedPassOptions<CreatePass>>
    : public basic_parser<llzk::NestedPassOptions<CreatePass>> {
public:
  using OptionsT = llzk::NestedPassOptions<CreatePass>;

  parser(Option &O) : basic_parser<OptionsT>(O) {}

  bool parse(Option &O, StringRef, StringRef Arg, OptionsT &Val) {
    StringRef options = Arg;
    if (options.consume_front("{")) {
      if (!options.consume_back("}")) {
        return O.error("expected nested pass options to end with '}'");
      }
    }

    auto pass = CreatePass();
    std::string error;
    if (failed(OptionsT::initializePass(*pass, options, error))) {
      return O.error(error);
    }

    Val.value = options.str();
    return false;
  }

  static void print(llvm::raw_ostream &OS, const OptionsT &Val) { OS << '{' << Val.value << '}'; }

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
