#ifndef LIB_DIALECT_ZKBUILDER_ZKBUILDERDIALECT_TD_
#define LIB_DIALECT_ZKBUILDER_ZKBUILDERDIALECT_TD_

include "mlir/IR/OpBase.td"

def ZKBuilder_Dialect : Dialect {
  let name = "ZKBuilder";
  let summary = "Dialect for ZKBuilder";
  let description = [{
    The ZKBuilder dialect defines types and operations for 
    ZK Builders from the ZK Lean project.
  }];

  let cppNamespace = "::mlir::zkbuilder";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_ZKBUILDER_ZKBUILDERDIALECT_TD_
