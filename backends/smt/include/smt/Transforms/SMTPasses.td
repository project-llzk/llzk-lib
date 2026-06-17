//===-- SMTPasses.td ----------------------------------------*- tablegen
//-*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SMT_PASSES_TD
#define SMT_PASSES_TD

include "llzk/Pass/PassBase.td"

def PrettyPrintSMTLIBPass : Pass<"smt-pretty-print", "::mlir::ModuleOp"> {
  let summary = "Pretty-print SMT dialect IR as SMTLIB";
  let description = [{
    Emits a textual SMTLIB script for entry-point SMT workflows. The pass
    supports the SMT subset currently produced by llzk's SMT lowering and
    verif-to-SMT pipeline.
  }];
  let dependentDialects = ["mlir::func::FuncDialect", "llzk::smt::SMTDialect",
                           "llzk::boolean::BoolDialect",
                           "mlir::arith::ArithDialect"];
  let options =
      [Option<
           "outputFilename", "output-file", "std::string",
           /*default=*/"\"-\"",
           "File to write the SMTLIB script to. Use '-' to write to stdout.">,
       Option<
           "entry", "entry", "std::string",
           /*default=*/"\"\"",
           "Export only the specified func.func root symbol. Defaults to all "
           "smt_verif_*_entry roots in source order.">,
       Option<
           "logic", "logic", "std::string",
           /*default=*/"\"ALL\"",
           "SMTLIB logic name to emit at the start of each exported script.">];
}

#endif // SMT_PASSES_TD
