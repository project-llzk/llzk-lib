//===-- Passes.td ------------------------------------------------*- td -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CONVERSIONS_PASSES_TD
#define LLZK_CONVERSIONS_PASSES_TD

include "llzk/Pass/PassBase.td"

def ConvertFeltConstToZKExprPass
    : Pass<"convert-felt-const-to-zkexpr", "::mlir::ModuleOp"> {
  let summary = "Wrap felt constants in ZKExpr.Literal ops";
  let description = [{
    Converts felt.const operations into the ZKExpr dialect by materializing a
    corresponding ZKExpr.Literal operation. The literal operation consumes a
    felt value so its operand is a cloned felt.const while the original result
    is replaced with the ZKExpr literal.
  }];
  let dependentDialects = [
    "llzk::felt::FeltDialect",
    "llzk::constrain::ConstrainDialect",
    "mlir::zkexpr::ZKExprDialect",
    "mlir::zkbuilder::ZKBuilderDialect"
  ];
  let constructor = "llzk::createConvertFeltConstToZKExprPass()";
}

#endif // LLZK_CONVERSIONS_PASSES_TD
