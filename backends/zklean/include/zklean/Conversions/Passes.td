//===-- Passes.td ------------------------------------------------*- td -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef ZKLEAN_CONVERSION_PASSES_TD
#define ZKLEAN_CONVERSION_PASSES_TD

include "llzk/Pass/PassBase.td"

def ConvertLLZKToZKLeanPass
    : Pass<"convert-llzk-to-zklean", "::mlir::ModuleOp"> {
  let summary = "Convert LLZK structs and constraints to ZKLean IR";
  let description = [{
    Converts an LLZK module into ZKLean dialects. The pass emits
    ZKLeanLean.structures from LLZK struct.defs, and
    rewrites constraint functions into ZKLeanLean functions.
    It wraps felt values in ZKExpr.Literal, lowers arithmetic to ZKExpr,
    constraints to ZKBuilder.ConstrainEq, struct member reads to ZKLeanLean.accessor, and
    maps bool.cmp/cast.tofelt into ZKLeanLean.call to stub functions.
  }];
  let dependentDialects = ["llzk::felt::FeltDialect",
                           "llzk::constrain::ConstrainDialect",
                           "llzk::zkexpr::ZKExprDialect",
                           "llzk::zkbuilder::ZKBuilderDialect",
                           "llzk::zkleanlean::ZKLeanLeanDialect"];
  let constructor = "zklean::createConvertLLZKToZKLeanPass()";
}

def ConvertZKLeanToLLZKPass
    : Pass<"convert-zklean-to-llzk", "::mlir::ModuleOp"> {
  let summary = "Convert ZKLean IR back to LLZK structs and constraints";
  let description = [{
    Converts ZKLean dialects back to LLZK. The pass materializes LLZK
    struct.defs from ZKLeanLean structures, rewrites ZKExpr arithmetic into
    felt operations, converts ZKBuilder constraints into constrain.eq ops,
    and lowers ZKLeanLean.accessor/call back into struct member reads and
    bool.cmp/cast.tofelt where applicable. It also emits minimal @compute stub
    or @constrain stubs when missing to satisfy LLZK invariants.
  }];
  let dependentDialects = ["llzk::felt::FeltDialect",
                           "llzk::constrain::ConstrainDialect",
                           "llzk::zkexpr::ZKExprDialect",
                           "llzk::zkbuilder::ZKBuilderDialect",
                           "llzk::zkleanlean::ZKLeanLeanDialect"];
  let constructor = "zklean::createConvertZKLeanToLLZKPass()";
}

#endif // ZKLEAN_CONVERSION_PASSES_TD
