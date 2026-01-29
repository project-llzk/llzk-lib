#ifndef LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANDIALECT_TD_
#define LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANDIALECT_TD_

include "mlir/IR/OpBase.td"

def ZKLeanLean_Dialect : Dialect {
  let name = "ZKLeanLean";
  let summary = "A dialect for ZKLean Lean primitives";
  let description = [{
    The ZKLeanLean dialect defines types and operations for Lean-facing
    primitives, including struct definitions, field access, and function calls.
  }];

  let cppNamespace = "::mlir::zkleanlean";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANDIALECT_TD_
