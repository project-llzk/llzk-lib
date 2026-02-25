//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PCL_CONV_TRANSFORMATION_PASSES_TD
#define PCL_CONV_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def PCLLoweringPass : LLZKPass<"llzk-to-pcl"> {
  let summary = "Rewrites constraints to be compatible with PCL constraints "
                "used by Picus";
  let description = [{
      Transforms LLZK constraints into an equivalent set of constraints expressed in the pcl-mlir dialect.
      This pass currently expects the flattening and inlining passes to be run prior to its use.
    }];
  let constructor = "pcl::conversion::createPCLLoweringPass()";
}

#endif // PCL_CONV_TRANSFORMATION_PASSES_TD
