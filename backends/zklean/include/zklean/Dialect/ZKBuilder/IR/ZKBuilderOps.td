//===-- ZKBuilderOps.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_TD_
#define LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_TD_

include "ZKBuilderDialect.td"
include "ZKBuilderTypes.td"

include "zklean/Dialect/ZKExpr/IR/ZKExprTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

////////////////////////////////////////////////////////
//        ZKBuilderOps
////////////////////////////////////////////////////////

def ZKBuilder_ConstrainEqOp : Op<ZKBuilder_Dialect, "ConstrainEq", []> {
  let summary = "Constrain two ZK Expressions to be equal";
  let arguments = (ins ZKExpr:$lhs, ZKExpr:$rhs);
  let results = (outs ZKBuilderState:$output);
  let assemblyFormat = "$lhs $rhs attr-dict";
}

def ZKBuilder_ConstrainR1CSOp : Op<ZKBuilder_Dialect, "ConstrainR1CS", []> {
  let summary = "Constrain R1CS in the Builder State";
  let arguments = (ins ZKExpr:$a, ZKExpr:$b, ZKExpr:$c);
  let results = (outs ZKBuilderState:$output);
  let assemblyFormat = "$a $b $c attr-dict";
}

def ZKBuilder_AllocWitnessOp
    : Op<ZKBuilder_Dialect, "AllocWitness", [ConstantLike]> {
  let summary = "Allocate a witness value";
  let results = (outs ZKExpr:$output);
  let assemblyFormat = "attr-dict";
}


#endif  // LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_TD_
