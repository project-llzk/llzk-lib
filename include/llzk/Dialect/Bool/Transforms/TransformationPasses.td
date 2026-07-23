//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_BOOLEAN_TRANSFORMATION_PASSES_TD
#define LLZK_BOOLEAN_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def LowerBoolQuantifiersPass : LLZKPass<"llzk-lower-bool-quantifiers"> {
  let summary = "Lower bool quantifiers to SCF loops";
  let description = [{
    Replace `bool.forall` and `bool.exists` over LLZK arrays with `scf.for`
    loops that iterate over the first array dimension and accumulate the
    predicate result with `bool.and` or `bool.or`, respectively.
  }];
}

#endif // LLZK_BOOLEAN_TRANSFORMATION_PASSES_TD
