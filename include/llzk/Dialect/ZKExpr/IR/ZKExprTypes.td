#ifndef LIB_DIALECT_ZKEXPR_ZKEXPRTYPES_TD_
#define LIB_DIALECT_ZKEXPR_ZKEXPRTYPES_TD_

include "llzk/Dialect/ZKExpr/IR/ZKExprDialect.td"
include "mlir/IR/AttrTypeBase.td"

// In zkExpr this is just an abbr for nat
def WitnessID: TypeDef<ZKExpr_Dialect, "WitnessID"> {
  let summary = "WitnessID";
  let mnemonic = "Witnessid";
  let description = [{
    WitnessID
  }];
}

def ZKExpr : TypeDef<ZKExpr_Dialect, "ZKExpr"> {
  let summary = "ZK Expressions";
  let mnemonic = "zkexpr";
  let description = [{
    A type for ZK Expressions.
  }];
}

def ComposedLookupTable : TypeDef<ZKExpr_Dialect, "ComposedLookupTable"> {
  let summary = "Lookup table parameterized by table dimensions";
  let mnemonic = "composedlookuptable";
  let description = [{
        Lookup table parameterized by table dimensions
  }];

  // TODO: See zkexpr/.../AST.expr and update params when clarified
  let parameters = (ins "int":$cols, "int":$rows);
  let assemblyFormat = "`<` $cols $rows `>`";
}

#endif  // LIB_DIALECT_ZKEXPR_ZKEXPRTYPES_TD_
