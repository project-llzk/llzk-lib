//===-- SymbolUseGraphPass.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-print-symbol-use-graph` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/AnalysisPasses.h"
#include "llzk/Analysis/SymbolUseGraph.h"

namespace llzk {
#define GEN_PASS_DEF_SYMBOLUSEGRAPHPRINTERPASS
#include "llzk/Analysis/AnalysisPasses.h.inc"
} // namespace llzk

using namespace llzk;

namespace {

class PassImpl : public llzk::impl::SymbolUseGraphPrinterPassBase<PassImpl> {
  using Base = SymbolUseGraphPrinterPassBase<PassImpl>;
  using Base::Base;

  void runOnOperation() override {
    markAllAnalysesPreserved();

    SymbolUseGraph &a = getAnalysis<SymbolUseGraph>();
    if (saveDotGraph) {
      a.dumpToDotFile();
    }
    a.print(toStream(outputStream));
  }
};

} // namespace
