//===-- ConversionPasses.td --------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PCL_CONV_CONVERSION_PASSES_TD
#define PCL_CONV_CONVERSION_PASSES_TD

include "llzk/Pass/PassBase.td"

def PCLLoweringPass : LLZKPass<"llzk-to-pcl"> {
  let summary =
      "Rewrite constraints to be compatible with PCL constraints used by Picus";
  let description = [{
      Transforms LLZK constraints into an equivalent set of constraints expressed in the pcl-mlir dialect.
      This pass currently expects the flattening and inlining passes to be run prior to its use.
    }];

  let options = [Option<
      "mode", "mode", "LlzkToPclMode", "::pcl::LlzkToPclMode::Full",
      "Lowering mode (full or stubbed)", [{::llvm::cl::values(
               clEnumValN(::pcl::LlzkToPclMode::Full, "full",
                "Lower the IR completely."),
               clEnumValN(::pcl::LlzkToPclMode::Stubbed, "stubbed",
                "Replace free functions containing unsupported ops with stubs.")
              )}]>];
}

#endif // PCL_CONV_CONVERSION_PASSES_TD
