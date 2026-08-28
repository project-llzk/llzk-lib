//===-- TransformationPasses.td ----------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_GLOBAL_TRANSFORMATION_PASSES_TD
#define LLZK_GLOBAL_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def ConstGlobalPropagationPass : LLZKPass<"llzk-const-global-propagation"> {
  let summary = "Propagate const globals into their use sites";
  let description = [{
    Propagate constant values in place of symbols that refer to `global.def`
    operations marked with the `const` attribute.
  }];
}

#endif // LLZK_GLOBAL_TRANSFORMATION_PASSES_TD
