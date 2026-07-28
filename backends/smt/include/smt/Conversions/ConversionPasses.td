//===-- ConversionPasses.td --------------------------------------*- td -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SMT_CONVERSIONPASSES_TD
#define SMT_CONVERSIONPASSES_TD

include "llzk/Pass/PassBase.td"

def SMTLoweringPass : LLZKPass<"llzk-to-smt-no-cf"> {
  let summary = "Lowers non-control flow LLZK ops to SMTLIB";
  let description = [{
    Lowers felt and array operations to their equivalents in
    the theory of integers and arrays, and generates
    assertions for struct member writes and constraints. This
    pass preserves control flow, and should be used in
    conjuction with the -llzk-to-smt-cf-only pass, or as
    part of the -llzk-to-smt pass pipeline.
  }];
  let options = [Option<"fieldName", "field", "std::string",
                        /* default */ "\"\"",
                        "Prime field name to use for SMT lowering. "
                        "If omitted, the pass auto-detects it.">];
}

def SMTNaiveLoweringPass : LLZKPass<"llzk-to-smt-no-cf-naive"> {
  let summary =
      "Lowers non-control flow LLZK ops to SMTLIB using naive mod p encoding";
  let description = [{
    Lowers felt operations to integer SMT operations using the original
    unoptimized encoding strategy, preserving explicit modulo reductions
    instead of exploiting interval-analysis results.
  }];
  let options = [Option<"fieldName", "field", "std::string",
                        /* default */ "\"\"",
                        "Prime field name to use for SMT lowering. "
                        "If omitted, the pass auto-detects it.">];
}

def SMTCFLoweringPass : LLZKPass<"llzk-to-smt-cf-only"> {
  let summary = "Lowers control-flow LLZK ops to SMTLIB";
  let description = [{
    Lowers scf.if by putting branching conditions into the
    antecedents of implications inside asserts, and guarding
    yielded values with smt.ite. This pass should come after
    the -llzk-to-smt-no-cf pass, or be used as part of the
    -llzk-to-smt pass pipeline.
  }];
}

#endif // SMT_CONVERSIONPASSES_TD
