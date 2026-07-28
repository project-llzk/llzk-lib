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
    Scalarize `pod.type` values by splitting POD-typed struct members into
    multiple scalar members, splitting POD-typed array elements into parallel
    arrays, then rewriting affected member accesses plus function signatures,
    calls, and returns, and finally running POD-specific SROA + mem2reg cleanup
    so the remaining POD storage is promoted to SSA values.

    If it is necessary to scalarize both PODs and arrays, run this pass before
    running the `-llzk-array-to-scalar` pass because that pass will not scalarize
    array types that are within a POD type.
  }];
}

#endif // LLZK_POD_TRANSFORMATION_PASSES_TD
