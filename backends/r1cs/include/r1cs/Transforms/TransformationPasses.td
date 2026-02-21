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
  let summary = "Rewrites constraints to be compatible with R1CS constraints "
                "i.e a*b - c = 0";
  let description = [{
    Transforms LLZK constraints into an equivalent set of R1CS constraints expressed in the r1cs dialect.
    This pass is best used as part of the `-llzk-full-r1cs-lowering` pipeline which includes
    a degree lowering pass and clean up passes to ensure correctness and performance.
  }];
  let constructor = "r1cs::createR1CSLoweringPass()";
}

#ifdef WITH_PCL
def PCLLoweringPass : LLZKPass<"llzk-to-pcl"> {
  let summary = "Rewrites constraints to be compatible with PCL constraints "
                "used by Picus";
  let description = [{
      Transforms LLZK constraints into an equivalent set of constraints expressed in the pcl-mlir dialect.
      This pass currently expects the flattening and inlining passes to be run prior to its use.
    }];
  let constructor = "r1cs::createPCLLoweringPass()";
  let options = [Option<
      "prime", "prime", "::llvm::APInt",
      "llvm::APInt(31, 2130706433)", // KoalaBear in decimal
      "The prime field that the constraints operate over. This "
      "is required by PCL, and the prime must be passed in digits.">];
}
#endif // WITH_PCL

#endif // R1CS_TRANSFORMATION_PASSES_TD
