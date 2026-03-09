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
    "Lowers felt and array operations to their equivalents in "
    "the theory of integers and arrays, and generates "
    "assertions for struct member writes and constraints. This "
    "pass preserves control flow, and should be used in "
    "conjuction with the -llzk-to-smt-cf-to-imp pass, or as "
    "part of the -llzk-to-smt pass pipeline."
  }];
  let constructor = "smt::createSMTLoweringPass()";
}

#endif // SMT_CONVERSIONPASSES_TD
