#ifndef LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_TD_
#define LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_TD_

include "ZKBuilderDialect.td"
include "ZKBuilderTypes.td"

include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

////////////////////////////////////////////////////////
//        ZKBuilderOps
////////////////////////////////////////////////////////

def ZKBuilder_ConstrainEqOp : Op<ZKBuilder_Dialect, "ConstrainEq", [Pure]> {
  let summary = "Constrain two ZK Expressions to be equal";
  let arguments = (ins ZKExpr:$lhs, ZKExpr:$rhs);
  let results = (outs ZKBuilderState:$output);
  let assemblyFormat = "$lhs $rhs attr-dict";
}

def ZKBuilder_ConstrainR1CSOp : Op<ZKBuilder_Dialect, "ConstrainR1CS", [Pure]> {
  let summary = "Constrain R1CS in the Builder State";
  let arguments = (ins ZKExpr:$a, ZKExpr:$b, ZKExpr:$c);
  let results = (outs ZKBuilderState:$output);
  let assemblyFormat = "$a $b $c attr-dict";
}

/*
def ZKBuilder_WitnessVarOp : Op<ZKBuilder_Dialect, "WitnessVar", [Pure, ConstantLike] {
  let summary = "Introduce ZK witness variable";
  let arguments = (ins AnyIntElementsAttr:$lit);
  let results = (outs ZKExpr:$output)
}
*/

#endif  // LIB_DIALECT_ZKBUILDER_ZKBUILDEROPS_TD_
