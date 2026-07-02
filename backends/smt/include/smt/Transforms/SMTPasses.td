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
    Emits an SMT-LIB script for the unique top-level smt.solver
    directly contained in the root builtin.module. Module-scope func.func
    definitions remain legal as helper code that may be called from within
    the smt.solver body.

    Expected input shape:

    ```mlir
    module {
      func.func @helper(%x: !smt.int) -> !smt.bool {
        %zero = smt.int.constant 0
        %isPos = smt.int.cmp gt %x, %zero
        return %isPos : !smt.bool
      }

      smt.solver () : () -> () {
        smt.set_info ":status" unsat
        smt.set_logic "ALL"
        %x = smt.declare_fun "x" : !smt.int
        %y = func.call @helper(%x) : (!smt.int) -> !smt.bool
        smt.assert %y
        smt.check sat { smt.yield } unknown { smt.yield } unsat { smt.yield }
        smt.yield
      }
    }
    ```

    Example output:

    ```smt2
    (set-info :status unsat)
    (set-logic ALL)
    (declare-fun x () Int)
    (define-fun helper ((helper__arg0 Int)) Bool
      (> helper__arg0 0))
    (assert (helper x))
    (check-sat)
    ```

    The resulting script can then be piped into `llzk-smt-check` or another
    SMT-LIB solver.

    Some SMT dialect ops with special handling:
    - If an `smt.set_logic` op is not present, the lowering emits a conservative
      `(set-logic ALL)` at the start of the script.
    - The `smt.check` operation in the SMT dialect supports conditional execution
      based on the solver result, which a SMT-LIB script cannot generally support.
      This pass therefore supports only bare checks with empty `sat`, `unknown`,
      and `unsat` regions, which lower to `(check-sat)`.
      If a lowering needs to branch on the solver result and emit different
      follow-up commands, it must target a higher-level driver representation
      outside SMT-LIB.
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
