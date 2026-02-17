//===-- ZKLeanPasses.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ZKLEAN_PASSES_TD
#define LLZK_ZKLEAN_PASSES_TD

include "llzk/Pass/PassBase.td"

def PrettyPrintZKLeanPass
    : Pass<"zklean-pretty-print", "::mlir::ModuleOp"> {
  let summary = "Pretty-print zkLean dialect IR as Lean code";
  let description = [{
    Emits a textual Lean-like rendering of ZKLean-related operations.
    The output is expected to be compilable by Lean4 with zkLean
  }];
  let dependentDialects = [
    "llzk::felt::FeltDialect",
    "mlir::func::FuncDialect",
    "mlir::zkexpr::ZKExprDialect",
    "mlir::zkbuilder::ZKBuilderDialect",
    "mlir::zkleanlean::ZKLeanLeanDialect"
  ];
  let options = [
    Option<"outputFilename", "output-file", "std::string",
           /* default */ "\"-\"",
           "File to write the pretty-printed zkLean dialect operations to. "
           "Use '-' to write to stdout.">,
  ];
  let constructor = "llzk::zklean::createPrettyPrintZKLeanPass()";
}

#endif // LLZK_ZKLEAN_PASSES_TD
