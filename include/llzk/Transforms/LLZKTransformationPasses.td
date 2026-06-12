//===-- LLZKTransformationPasses.td ------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_TRANSFORMATION_PASSES_TD
#define LLZK_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"

def RedundantReadAndWriteEliminationPass
    : LLZKPass<"llzk-duplicate-read-write-elim"> {
  let summary = "Remove redundant reads and writes";
  let description = [{
    Remove read and write operations to struct members and arrays that are redundant or unnecessary.
  }];
}

def RedundantOperationEliminationPass : LLZKPass<"llzk-duplicate-op-elim"> {
  let summary = "Remove redundant operations";
  let description = [{
    Remove llzk and arith dialect operations that produce the same results
    as previously executed operations.

    Pass should be run after `llzk-duplicate-read-write-elim` for maximum effect.
  }];
}

def UnusedDeclarationEliminationPass
    : LLZKPass<"llzk-unused-declaration-elim"> {
  let summary = "Remove unused member and struct declarations";
  let description = [{
    Remove member and struct declarations that are unused within the current compilation
    unit. Note that this pass may cause linking issues with external modules that
    depend on any unused member and struct declarations from this compilation unit.

    Pass should be run after `llzk-duplicate-read-write-elim`
    and `llzk-duplicate-op-elim` for maximum effect.
  }];
  let options = [Option<"removeStructs", "remove-structs", "bool",
                        /* default */ "false",
                        "Whether to remove unused struct definitions as well. "
                        "Requires module to declare a Main component, "
                        "otherwise all components will appear unused.">,
  ];
}

def RemoveUnusedDiscardableAllocationsPass
    : LLZKPass<"llzk-remove-unused-discardable-allocations"> {
  let summary = "Remove unread discardable allocations and their dead stores";
  let description = [{
    Remove ops with the selected allocator operation name when they are marked with
    `MemAlloc<DiscardableAllocationResource>`, the allocation has no reads, and every direct user is
    a discardable allocation accessor that can be erased as a dead store.
  }];
  let options = [Option<
      "allocatorOpName", "allocator-op", "std::string",
      /* default */ "\"\"",
      "Operation name of the discardable allocator to remove">];
}

def PolyLoweringPass : LLZKPass<"llzk-poly-lowering-pass"> {
  let summary =
      "Lower the degree of all polynomial equations to a specified maximum";
  let description = [{
    Rewrites constraint expressions into an (observationally) equivalent system where the degree of
    every polynomial is less than or equal to the specified maximum.

    High-degree subexpressions are factored into auxiliary struct members. The pass also recurses
    through degree-neutral roots such as `felt.add`, `felt.sub`, and `felt.neg`, so composite
    equality operands and struct constrain call arguments satisfy their degree bounds.

    This pass is best used as part of the `-llzk-full-poly-lowering` pipeline, which includes
    additional cleanup passes to ensure correctness and optimal performance.
  }];
  let options = [Option<"maxDegree", "max-degree", "unsigned",
                        /* default */ "2",
                        "Maximum degree of constraint polynomials "
                        "(default 2, minimum 2)">,
  ];
}

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
                        /* default */ "0",
                        "Maximum allowed constraint+multiplications in merged "
                        "@constrain functions">,
  ];
}

def ComputeConstrainToProductPass
    : LLZKPass<"llzk-compute-constrain-to-product"> {
  let summary = "Replace separate @compute and @constrain functions in a "
                "struct with a single @product function";
  let description = summary;
  let options = [Option<"rootStruct", "root-struct", "std::string",
                        /* default */ "\"@Main\"",
                        "Root struct at which to start alignment "
                        "(default to `@Main`)">];
}

def FuseProductLoopsPass : LLZKPass<"llzk-fuse-product-loops"> {
  let summary = "Fuse matching witness/constraint loops in a @product function";
  let description = summary;
}

def EnforceNoMemberOverwritePass : LLZKPass<"llzk-enforce-no-overwrite"> {
  let summary = "Checks that every struct member is written exactly once";
  let description = [{
      This pass currently reports an error if any struct member may not be written
      exactly once (i.e. overwritten or left uninitialized), and does not
      attempt to perform any repairs (e.g. SSA-ifying overwritten struct
      members, or default-initializing unwritten members). This pass
      overapproximates conditionals, and can result in false positives.
  }];
}

def WhileToForPass : LLZKPass<"llzk-while-to-for"> {
  let summary =
      "Converts scf.while loops to equivalent scf.for loops when possible";
  let description =
      "This pass identifies scf.while loops that have an "
      "induction variable and a uniform step, and converts them "
      "to equivalent scf.for loops, which are preferred by some "
      "analyses. This pass may introduce some spurious felt/index casts which "
      "should be cleaned up by --canonicalize.";
}

#endif // LLZK_TRANSFORMATION_PASSES_TD
