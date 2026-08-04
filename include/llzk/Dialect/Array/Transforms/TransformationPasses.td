//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ARRAY_TRANSFORMATION_PASSES_TD
#define LLZK_ARRAY_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def ArrayToScalarPass : LLZKPass<"llzk-array-to-scalar"> {
  let summary = "Replace arrays with scalar values";
  let description = [{
    Replace known-shape arrays with the proper number of scalar values
  }];
}

#endif // LLZK_ARRAY_TRANSFORMATION_PASSES_TD
