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

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/CommandLine.h>

// Custom command line parsers
namespace llvm {
namespace cl {

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
    assert(std::cmp_less_equal(Arg.size(), std::numeric_limits<unsigned>::max()));
    unsigned bits = std::max(1u, 4u * (unsigned)Arg.size());
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
      const Option &O, const APInt &V, OptionValue<APInt> Default, size_t GlobalWidth
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
