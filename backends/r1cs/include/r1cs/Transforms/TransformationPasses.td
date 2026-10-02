//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef R1CS_TRANSFORMATION_PASSES_TD
#define R1CS_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def R1CSLoweringPass : LLZKPass<"llzk-r1cs-lowering"> {
  let summary = "Lower LLZK constraints to R1CS";
  let description = [{
    By default, requires `poly.evaluated_main` and a straight-line main constrain
    function with its original aggregate storage interface. Run
    `llzk-monomorphize` and `llzk-evaluate-constraints` first. Do not pre-flatten
    evaluated input. Preserves structs and rolled compute functions, emits a
    circuit with `poly.wire_bindings`, and records it in `r1cs.main`. Repeated
    direct emission is a no-op.

    With `legacy=true`, accepts flattened, straight-line constrain bodies and
    replaces structs with circuits. Evaluated modules are rejected in legacy
    mode. Nested-region, successor-bearing, and multi-block constraints must be
    lowered before emission. Both modes prepare constraints as needed; use
    `llzk-full-r1cs-lowering` with the same legacy option for degree lowering.
  }];
  let options = [Option<"legacy", "legacy", "bool", "false",
                        "Use legacy flattened-input lowering instead of direct "
                        "storage lowering">];
}

def R1CSPreparePass : LLZKPass<"llzk-r1cs-prepare"> {
  let summary = "Normalize constraints and materialize R1CS auxiliary witness "
                "assignments";
  let description = [{
    Requires straight-line constrain bodies. Normalizes constraints, adds auxiliary
    members and their compute assignments, and marks each prepared struct with
    `r1cs.prepared`. Does not emit circuits or erase structs. Repeated preparation
    is a no-op. On evaluated modules, prepares only the main struct; otherwise
    prepares every struct. Shared by both lowering modes.
  }];
}

#endif // R1CS_TRANSFORMATION_PASSES_TD
