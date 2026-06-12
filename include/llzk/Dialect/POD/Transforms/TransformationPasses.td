//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_TRANSFORMATION_PASSES_TD
#define LLZK_POD_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def PodToScalarPass : LLZKPass<"llzk-pod-to-scalar"> {
  let summary = "Replace PODs with scalar values";
  let description = [{
    Replace `pod.type` values with the proper number of scalar values
  }];
}

#endif // LLZK_POD_TRANSFORMATION_PASSES_TD
