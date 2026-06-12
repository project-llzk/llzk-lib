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
    This pass is best used as part of the `-llzk-full-r1cs-lowering` pipeline which includes
    a degree lowering pass and clean up passes to ensure correctness and performance.
  }];
  let constructor = "r1cs::createR1CSLoweringPass()";
}

def R1CSBinaryExportPass : Pass<"r1cs-export-binary", "::mlir::ModuleOp"> {
  let summary = "Export an R1CS circuit to the standard .r1cs binary format";
  let description = [{
    Validates and exports one `r1cs.circuit` from the input module into the
    standard `.r1cs` binary container.
  }];
  let dependentDialects = ["r1cs::R1CSDialect"];
  let options =
      [Option<"outputFilename", "output-file", "std::string", "\"\"",
              "File to write the binary .r1cs output to.">,
       Option<"prime", "prime", "std::string", "\"\"",
              "Prime modulus for the field, encoded as a base-10 integer.">,
       Option<"circuitName", "circuit-name", "std::string", "\"\"",
              "Optional circuit symbol name to export when the module contains "
              "multiple r1cs.circuit ops.">];
  let constructor = "r1cs::createR1CSBinaryExportPass()";
}

#endif // R1CS_TRANSFORMATION_PASSES_TD
