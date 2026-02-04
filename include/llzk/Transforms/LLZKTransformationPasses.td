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
  let constructor = "llzk::createRedundantReadAndWriteEliminationPass()";
}

def RedundantOperationEliminationPass : LLZKPass<"llzk-duplicate-op-elim"> {
  let summary = "Remove redundant operations";
  let description = [{
    Remove llzk and arith dialect operations that produce the same results
    as previously executed operations.

    Pass should be run after `llzk-duplicate-read-write-elim` for maximum effect.
  }];
  let constructor = "llzk::createRedundantOperationEliminationPass()";
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
  let constructor = "llzk::createUnusedDeclarationEliminationPass()";
  let options = [Option<"removeStructs", "remove-structs", "bool",
                        /* default */ "false",
                        "Whether to remove unused struct definitions as well. "
                        "Requires module to declare a Main component, "
                        "otherwise all components will appear unused.">,
  ];
}

def PolyLoweringPass : LLZKPass<"llzk-poly-lowering-pass"> {
  let summary =
      "Lowers the degree of all polynomial equations to a specified maximum";
  let description = [{
    Rewrites constraint expressions into an (observationally) equivalent system where the degree of
    every polynomial is less than or equal to the specified maximum.

    This pass is best used as part of the `-llzk-full-poly-lowering` pipeline, which includes
    additional cleanup passes to ensure correctness and optimal performance.
  }];
  let constructor = "llzk::createPolyLoweringPass()";
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
  let constructor = "llzk::createInlineStructsPass()";
  let options = [Option<"maxComplexity", "max-merge-complexity", "uint64_t",
                        /* default */ "0",
                        "Maximum allowed constraint+multiplications in merged "
                        "@constrain functions">,
  ];
}

def R1CSLoweringPass : LLZKPass<"llzk-r1cs-lowering"> {
  let summary = "Rewrites constraints to be compatible with R1CS constraints "
                "i.e a*b - c = 0";
  let description = [{
    Transforms LLZK constraints into an equivalent set of R1CS constraints expressed in the r1cs dialect.
    This pass is best used as part of the `-llzk-full-r1cs-lowering` pipeline which includes
    a degree lowering pass and clean up passes to ensure correctness and performance.
  }];
  let constructor = "llzk::createR1CSLoweringPass()";
}

def ComputeConstrainToProductPass
    : LLZKPass<"llzk-compute-constrain-to-product"> {
  let summary = "Replaces separate @compute and @constrain functions in a "
                "struct with a single @product function";
  let description = summary;
  let constructor = "llzk::createComputeConstrainToProductPass()";
  let options = [Option<"rootStruct", "root-struct", "std::string",
                        /* default */ "\"@Main\"",
                        "Root struct at which to start alignment "
                        "(default to `@Main`)">];
}

def FuseProductLoopsPass : LLZKPass<"llzk-fuse-product-loops"> {
  let summary =
      "Fuses matching witness/constraint loops in a @product function";
  let description = summary;
  let constructor = "llzk::createFuseProductLoopsPass()";
  let options = [];
}

#ifdef WITH_PCL
def PCLLoweringPass : LLZKPass<"llzk-to-pcl"> {
  let summary = "Rewrites constraints to be compatible with PCL constraints "
                "used by Picus";
  let description = [{
      Transforms LLZK constraints into an equivalent set of constraints expressed in the pcl-mlir dialect.
      This pass currently expects the flattening and inlining passes to be run prior to its use.
    }];
  let constructor = "llzk::createPCLLoweringPass()";
  let options = [Option<
      "prime", "prime", "::llvm::APInt",
      "llvm::APInt(31, 2130706433)", // KoalaBear in decimal
      "The prime field that the constraints operate over. This "
      "is required by PCL, and the prime must be passed in digits.">];
}
#endif // WITH_PCL

#endif // LLZK_TRANSFORMATION_PASSES_TD
