#ifndef LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTDIALECT_TD_
#define LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTDIALECT_TD_

include "mlir/IR/OpBase.td"

def ZKLeanStruct_Dialect : Dialect {
  let name = "ZKLeanStruct";
  let summary = "A dialect for ZKLean struct definitions";
  let description = [{
    The ZKLeanStruct dialect defines types and operations for ZKLean
    structure definitions and field access.
  }];

  let cppNamespace = "::mlir::zkleanstruct";

  let useDefaultTypePrinterParser = 1;
}

#endif  // LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTDIALECT_TD_
