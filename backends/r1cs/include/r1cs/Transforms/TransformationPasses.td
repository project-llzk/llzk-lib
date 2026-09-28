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
  let summary = "Rewrite constraints to be compatible with R1CS constraints, "
                "i.e a*b - c = 0";
  let description = [{
    Transforms LLZK constraints into an equivalent set of R1CS constraints expressed in the r1cs dialect.
    The pass expects already-flattened, straight-line `constrain` bodies. Run `llzk-flatten` or
    another control-flow lowering pass before this pass; nested-region, successor-bearing, or
    multi-block constraints are rejected before R1CS normalization.
    This pass is best used on already-flattened input as part of the `-llzk-full-r1cs-lowering`
    pipeline which includes a degree lowering pass and clean up passes to ensure correctness and
    performance.
  }];
}

def R1CSPreparePass : LLZKPass<"llzk-r1cs-prepare"> {
  let summary = "Normalize constraints and materialize R1CS auxiliary witness "
                "assignments";
  let description = [{
    Requires straight-line constrain bodies. Normalizes constraints, adds auxiliary
    members and their compute assignments, and marks each prepared struct with
    `r1cs.prepared`. Does not emit circuits or erase structs. Repeated preparation
    is a no-op. On evaluated modules, prepares only the main struct; otherwise
    prepares every struct. Shared by the legacy and direct lowering passes.
  }];
}

def R1CSDirectLoweringPass : LLZKPass<"llzk-r1cs-direct-lowering"> {
  let summary = "Lower evaluated storage constraints directly to R1CS";
  let description = [{
    Requires `poly.evaluated_main` and an evaluated, straight-line main constrain
    function with its original aggregate storage interface. Do not pre-flatten
    evaluated input. Prepares constraints as needed and operates on main only.
    Preserves structs and rolled compute functions, emits a circuit with
    `poly.wire_bindings`, and records it in `r1cs.main`. Repeated emission is a
    no-op. Use `llzk-full-direct-r1cs-lowering` to include degree lowering.
    Non-evaluated modules are rejected; use `llzk-r1cs-lowering` for legacy input.
  }];
}

#endif // R1CS_TRANSFORMATION_PASSES_TD
