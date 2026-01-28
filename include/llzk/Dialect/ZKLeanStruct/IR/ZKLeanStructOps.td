#ifndef LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTOPS_TD_
#define LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTOPS_TD_

include "ZKLeanStructDialect.td"
include "ZKLeanStructTypes.td"
include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ZKLeanStructOp<string mnemonic, list<Trait> traits = []>
    : Op<ZKLeanStruct_Dialect, mnemonic, traits>;

//===------------------------------------------------------------------===//
// Struct definitions
//===------------------------------------------------------------------===//

def ZKLeanStruct_StructDefOp
    : ZKLeanStructOp<
          "def", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                  IsolatedFromAbove,
                  NoRegionArguments]#GraphRegionNoTerminator.traits> {
  let summary = "ZKLean struct definition";
  let description = [{
    Defines a ZKLean struct with named fields. Only field definitions are
    allowed in the body region.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{ $sym_name $bodyRegion attr-dict }];
}

def ZKLeanStruct_FieldDefOp
    : ZKLeanStructOp<"field", [HasParent<"::mlir::zkleanstruct::StructDefOp">,
                               Symbol]> {
  let summary = "ZKLean struct field definition";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<ZKExpr>:$type);
  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];
}

//===------------------------------------------------------------------===//
// Struct operations
//===------------------------------------------------------------------===//

def ZKLeanStruct_AccessorOp : ZKLeanStructOp<"accessor", [Pure]> {
  let summary = "access value of a struct field";
  let arguments = (ins ZKLeanStruct_StructType:$component,
                       FlatSymbolRefAttr:$field_name);
  let results = (outs ZKExpr:$value);
  let assemblyFormat = [{
    $component `[` $field_name `]` `:` type($component) `,` type($value) attr-dict
  }];
}

#endif  // LIB_DIALECT_ZKLEANSTRUCT_ZKLEANSTRUCTOPS_TD_
