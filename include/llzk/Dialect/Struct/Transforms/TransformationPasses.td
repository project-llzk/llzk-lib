//===-- TransformationPasses.td ------------------------------*- tablegen
//-*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DIALECT_STRUCT_TRANSFORMS_TRANSFORMATION_PASSES_TD
#define LLZK_DIALECT_STRUCT_TRANSFORMS_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def InlineStructsPass : LLZKPass<"llzk-inline-structs"> {
  let summary = "Inlines nested structs (i.e., subcomponents).";
  let description = [{
    This pass inlines nested structs (i.e., subcomponents) at struct-type members and at calls to the
    subcomponent compute/constrain functions. Inlining decisions are guided by the call graph of
    "constrain" functions.

    The `max-merge-complexity` parameter can be used to limit the complexity of the resulting structs such
    that a potential inlining will not take place if doing so would push the sum of constraint and
    multiplications in the combined struct over the limit. The default value `0` indicates no limits
    which means all structs will be inlined into the Main struct.

    This pass should be run after `llzk-flatten` to ensure structs do not have template parameters
    because structs with template parameters cannot (currently) be inlined. Inlining is also not
    (currently) supported for subcomponent structs stored in an array-type member.

    This pass also assumes that all subcomponents that are created by calling a struct "@compute"
    function are ultimately written to exactly one member within the current struct.
  }];
  let options = [Option<"maxComplexity", "max-merge-complexity", "uint64_t",
                        /* default: no limit */ "0",
                        "Maximum allowed constraint+multiplications in merged "
                        "@constrain functions">,
  ];
}

#endif // LLZK_DIALECT_STRUCT_TRANSFORMS_TRANSFORMATION_PASSES_TD
