//===-- ZKExprTypes.td -------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_ZKEXPR_ZKEXPRTYPES_TD_
#define LIB_DIALECT_ZKEXPR_ZKEXPRTYPES_TD_

include "zklean/Dialect/ZKExpr/IR/ZKExprDialect.td"
include "mlir/IR/AttrTypeBase.td"

// In zkExpr this is just an abbr for nat
def WitnessID: TypeDef<ZKExpr_Dialect, "WitnessID"> {
  let summary = "WitnessID";
  let mnemonic = "Witnessid";
}

def ZKExpr : TypeDef<ZKExpr_Dialect, "ZKExpr"> {
  let summary = "ZK Expressions";
  let mnemonic = "zkexpr";
}

def ComposedLookupTable : TypeDef<ZKExpr_Dialect, "ComposedLookupTable"> {
  let summary = "Lookup table parameterized by table dimensions";
  let mnemonic = "composedlookuptable";

  // TODO: See zkexpr/.../AST.expr and update params when clarified
  let parameters = (ins "int":$cols, "int":$rows);
  let assemblyFormat = "`<` $cols $rows `>`";
}

#endif  // LIB_DIALECT_ZKEXPR_ZKEXPRTYPES_TD_
