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
  let summary = "Convert flattened LLZK constraints to R1CS";
  let options = [Option<"outputLLZK", "output-llzk", "std::string", "\"\"",
                        "If non-empty, write LLZK with auxiliary fields and "
                        "assignments in compute to "
                        "this file">,
                 Option<"outputLayout", "output-layout", "std::string", "\"\"",
                        "If non-empty, write access paths, LLZK layout IDs, "
                        "and R1CS wire IDs "
                        "as CSV to this file">];
  let description = [{
    Converts the constraints in llzk.main to R1CS. Run
    `llzk-constraint-flattening` and `llzk-poly-lowering-pass` with
    `main-only=true max-degree=2` first. The
    `llzk-constraint-flattening-r1cs-lowering` pipeline runs these passes together.

    The pass performs the following steps:

    1. Check that every constraint is polynomial and has degree at most two.
    2. Add auxiliary signal fields and equations where needed to express each
       constraint as A * B = C. Add assignments in @compute for those fields.
    3. Compute the LLZK layout, including the auxiliary signals.
    4. Assign R1CS wire IDs: zero for constant one, then public outputs, public
       inputs, private inputs, and internal signals.
    5. Create the R1CS wires and convert the constraints to use them.

    The resulting module contains R1CS IR. Set `output-llzk=<file>` to write
    LLZK with auxiliary fields and assignments in @compute to that file. Set
    `output-layout=<file>` to write a CSV with columns access_path,
    llzk_layout_id, and r1cs_wire_id to that file. Each option defaults to an
    empty string, which disables its file output.

    LLZK layout IDs and R1CS wire IDs may differ. Each CSV row maps the same
    signal, identified by its access path, between the two. The CSV is written
    directly from the layout and wire map built during lowering; no LLZK layout
    information is stored in the R1CS IR.
  }];
}

#endif // R1CS_TRANSFORMATION_PASSES_TD
