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

    Expected input shape:

    ```mlir
    module {
      func.func @helper(%x: !smt.int) -> !smt.bool {
        %zero = smt.int.constant 0
        %isPos = smt.int.cmp gt %x, %zero
        return %isPos : !smt.bool
      }

      smt.solver () : () -> () {
        smt.set_logic "ALL"
        %x = smt.declare_fun "x" : !smt.int
        %y = func.call @helper(%x) : (!smt.int) -> !smt.bool
        smt.assert %y
        smt.yield
      }
    }
    ```

    Logic handling is intentionally simple. If the selected solver body contains
    an `smt.set_logic`, the exporter preserves it exactly where it appears. If
    the IR does not specify a logic, the exporter emits the conservative SMT-LIB
    fallback `(set-logic ALL)` at the start of the script instead of trying to
    infer a narrower theory combination.

    Example output:

    ```smt2
    (set-logic ALL)
    (set-info :llzk-root "module")
    (declare-fun x () Int)
    (define-fun helper ((helper__arg0 Int)) Bool
      (> helper__arg0 0))
    (assert (helper x))
    ```

    The resulting script can be piped directly into `llzk-smt-check` for staged
    result verification, or into a general SMT solver such as `z3` when only the
    exported SMT-LIB script is needed.
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
