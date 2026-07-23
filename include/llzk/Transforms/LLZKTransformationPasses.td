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
    Remove redundant or unnecessary reads and writes to struct members, arrays,
    globals, and RAM. Global writes are removed when they are overwritten or
    write an already-known value. RAM stores require the same translated address
    SSA value.

    Struct-member accesses match only when the base, member, table offset,
    affine map metadata, and translated affine operands match. Omitted
    member-read offsets and member writes denote the current row.

    Stateful reads are reused when the current analysis state proves the read
    observes the same value. Writes and stores update that known state, while
    unknown or non-read state effects invalidate it.
  }];
}

def RedundantOperationEliminationPass : LLZKPass<"llzk-duplicate-op-elim"> {
  let summary = "Remove redundant operations";
  let description = [{
    Remove equivalent memory-effect-free operations and duplicate LLZK constraints.

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

    The pass expects already-flattened, straight-line `compute` and `constrain` bodies. Run
    `llzk-flatten` or another control-flow lowering pass before this pass; nested-region,
    successor-bearing, or multi-block functions are rejected instead of being rewritten with
    component-lifetime auxiliary members.

    This pass is best used on already-flattened input as part of the `-llzk-full-poly-lowering`
    pipeline, which includes additional cleanup passes to ensure correctness and optimal
    performance.
  }];
  let options = [Option<"maxDegree", "max-degree", "unsigned",
                        /* default */ "2",
                        "Maximum degree of constraint polynomials "
                        "(default 2, minimum 2)">,
  ];
}

def ComputeConstrainToProductPass
    : LLZKPass<"llzk-compute-constrain-to-product"> {
  let summary = "Replace separate @compute and @constrain functions in a "
                "struct with a single @product function";
  let description = summary;
  let options = [Option<"rootStruct", "root-struct", "std::string",
                        /* default */ "\"Main\"",
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
