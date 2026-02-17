//===-- Passes.td ------------------------------------------------*- td -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONVERSIONS_PASSES_TD
#define LLZK_CONVERSIONS_PASSES_TD

include "llzk/Pass/PassBase.td"

def ConvertLLZKToZKLeanPass
    : Pass<"convert-llzk-to-zklean", "::mlir::ModuleOp"> {
  let summary = "Wrap felt constants in ZKExpr.Literal ops";
  let description = [{
    Converts felt and constrain operations into the ZKLean dialects by
    materializing corresponding ZKExpr.Literal for each felt.const operation
    and replacing felt / constrain operations with respective ZKExpr /
    ZKBuilder operations. The literal operation consumes a felt value so its
    operand is a cloned felt.const while the original result is replaced with
    the ZKExpr literal.
  }];
  let dependentDialects = [
    "llzk::felt::FeltDialect",
    "llzk::constrain::ConstrainDialect",
    "llzk::zkexpr::ZKExprDialect",
    "llzk::zkbuilder::ZKBuilderDialect"
  ];
  let constructor = "llzk::createConvertLLZKToZKLeanPass()";
}

def ConvertZKLeanToLLZKPass
    : Pass<"convert-zklean-to-llzk", "::mlir::ModuleOp"> {
  let summary = "Convert ZKExpr/ZKBuilder ops back to felt/constrain dialect";
  let description = [{
    Converts the lean IR back to LLZK-style felt operations by translating
    ZKExpr arithmetic into felt arithmetic and ZKBuilder constraints into
    constrain.eq operations.
  }];
  let dependentDialects = [
    "llzk::felt::FeltDialect",
    "llzk::constrain::ConstrainDialect",
    "llzk::zkexpr::ZKExprDialect",
    "llzk::zkbuilder::ZKBuilderDialect"
  ];
  let constructor = "llzk::createConvertZKLeanToLLZKPass()";
}

#endif // LLZK_CONVERSIONS_PASSES_TD
