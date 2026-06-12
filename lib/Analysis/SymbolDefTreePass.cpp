//===-- SymbolDefTreePass.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-symbol-def-tree` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/SymbolDefTree.h"

namespace llzk {
#define GEN_PASS_DEF_SYMBOLDEFTREEPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"
} // namespace llzk

using namespace llzk;

namespace {

class PassImpl : public llzk::impl::SymbolDefTreePrinterPassBase<PassImpl> {
  using Base = SymbolDefTreePrinterPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    markAllAnalysesPreserved();

    SymbolDefTree &a = getAnalysis<SymbolDefTree>();
    if (saveDotGraph) {
      a.dumpToDotFile();
    }
    a.print(toStream(outputStream));
  }
};

} // namespace
