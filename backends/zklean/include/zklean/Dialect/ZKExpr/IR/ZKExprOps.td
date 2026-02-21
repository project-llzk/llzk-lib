//===-- ZKExprOps.td ---------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//


#ifndef LIB_DIALECT_ZKEXPR_ZKEXPROPS_TD_
#define LIB_DIALECT_ZKEXPR_ZKEXPROPS_TD_

include "ZKExprDialect.td"
include "ZKExprTypes.td"
include "llzk/Dialect/Felt/IR/Types.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ZKExpr_BinOp<string mnemonic> : Op<ZKExpr_Dialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
  let arguments = (ins ZKExpr:$lhs, ZKExpr:$rhs);
  let results = (outs ZKExpr:$output);
  let assemblyFormat = "$lhs $rhs attr-dict";
}

def ZKExpr_AddOp : ZKExpr_BinOp<"Add"> {
  let summary = "Addition operation between ZK Expressions.";
}

def ZKExpr_SubOp : ZKExpr_BinOp<"Sub"> {
  let summary = "Subtraction operation between ZK Expressions.";
}

def ZKExpr_MulOp : ZKExpr_BinOp<"Mul"> {
  let summary = "Multiplication operation between ZK Expressions.";
}

def ZKExpr_NegOp : Op<ZKExpr_Dialect, "Neg", [Pure]> {
  let summary = "Negate a ZK Expression";
  let arguments = (ins ZKExpr:$value);
  let results = (outs ZKExpr:$output);
  let assemblyFormat = "$value attr-dict";
}

// Note: This takes as input an LLZK Felt type
def ZKExpr_LiteralOp : Op<ZKExpr_Dialect, "Literal", [Pure]> {
  let summary = "Introduce ZK literal";
  let arguments = (ins LLZK_FeltType:$literal);
  let results = (outs ZKExpr:$output);
  let assemblyFormat = "$literal attr-dict";
}

#endif  // LIB_DIALECT_ZKEXPR_ZKEXPROPS_TD_
