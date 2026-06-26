//===-- SMTPasses.td ---------------------------------------*- tablegen -*-===//
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

def SMTDialectToSMTLIBPass : Pass<"smt-to-smtlib", "::mlir::ModuleOp"> {
  let summary = "Convert SMT dialect IR to SMT-LIB";
  let description = [{
    Emits a textual SMT-LIB script for an explicitly selected SMT root
    function. The pass supports the SMT subset currently produced by llzk's
    lowering pipelines.
  }];
  let dependentDialects = ["mlir::func::FuncDialect", "llzk::smt::SMTDialect",
                           "llzk::boolean::BoolDialect",
                           "mlir::arith::ArithDialect"];
  let options =
      [Option<
           "outputFilename", "output-file", "std::string",
           /*default=*/"\"-\"",
           "File to write the SMT-LIB script to. Use '-' to write to stdout.">,
       Option<"entry", "entry", "std::string",
              /*default=*/"\"\"",
              "Export the specified func.func root symbol. This option is "
              "required.">,
       Option<
           "logic", "logic", "std::string",
           /*default=*/"\"ALL\"",
           "SMT-LIB logic name to emit at the start of each exported script.">];
}

#endif // SMT_PASSES_TD
