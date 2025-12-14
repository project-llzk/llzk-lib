#ifndef LIB_DIALECT_ZKBUILDER_ZKBUILDERTYPES_TD_
#define LIB_DIALECT_ZKBUILDER_ZKBUILDERTYPES_TD_

include "ZKBuilderDialect.td"
include "mlir/IR/AttrTypeBase.td"

def ZKBuilderState : TypeDef<ZKBuilder_Dialect, "ZKBuilderState"> {
  let summary = "State of the ZKBuilder";
  let mnemonic = "zkbuilderstate";
  let description = [{
        State of the ZKBuilder
  }];
}

#endif  // LIB_DIALECT_ZKBUILDER_ZKBUILDERTYPES_TD_
