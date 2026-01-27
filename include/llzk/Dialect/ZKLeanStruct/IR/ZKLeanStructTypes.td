#ifndef LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTTYPES_TD_
#define LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTTYPES_TD_

include "ZKLeanStructDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"

// ZKLean struct instance type.
def ZKLeanStruct_StructType : TypeDef<ZKLeanStruct_Dialect, "Struct"> {
  let summary = "ZKLean struct value type";
  let mnemonic = "type";
  let description = [{
    Type of a ZKLean struct instance. The referenced struct definition is
    identified by a symbol reference.
  }];

  let parameters =
      (ins TypeParameter<
           "::mlir::SymbolRefAttr",
           "Fully-qualified name of the struct definition.">:$nameRef);
  let assemblyFormat = "`<` $nameRef `>`";
}

#endif  // LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTTYPES_TD_
