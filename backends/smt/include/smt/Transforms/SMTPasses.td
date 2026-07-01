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
    Emits a textual SMT-LIB script for the unique top-level smt.solver
    directly contained in the root builtin.module. Module-scope func.func
    definitions remain legal as helper code that may be called from within
    the smt.solver body.
  }];
  let dependentDialects = ["mlir::func::FuncDialect", "llzk::smt::SMTDialect",
                           "llzk::boolean::BoolDialect",
                           "mlir::arith::ArithDialect"];
  let options = [Option<
      "outputFilename", "output-file", "std::string",
      /*default=*/"\"-\"",
      "File to write the SMT-LIB script to. Use '-' to write to stdout.">];
}

#endif // SMT_PASSES_TD
