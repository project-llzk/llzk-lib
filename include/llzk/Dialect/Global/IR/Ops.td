//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_GLOBAL_OPS
#define LLZK_GLOBAL_OPS

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Global/IR/Dialect.td"
include "llzk/Dialect/Global/IR/OpInterfaces.td"
include "llzk/Dialect/Function/IR/OpTraits.td"

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class GlobalDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<GlobalDialect, mnemonic, traits>;

def LLZK_GlobalDefOp
    : GlobalDialectOp<"def", [HasParent<"mlir::ModuleOp">,
                              DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                              Symbol]> {
  let summary = "global value";
  let description = [{
    Examples:

    ```llzk
    // Global constant (denoted by "const" modifier) string.
    global.def const @s : !string.type = "Hello World!"

    // Global variable (i.e., no "const" modifier) with initial value.
    global.def @b : i1 = false

    // Uninitialized global variable.
    global.def @a : !array.type<2,2 x i1>
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, UnitAttr:$constant,
      TypeAttrOf<GlobalDefType>:$type,
      DefaultValuedAttr<AnyAttr, "nullptr">:$initial_value);

  let assemblyFormat = [{
    (`const` $constant^)?
    $sym_name `:` $type
    `` custom<GlobalInitialValue>($initial_value, ref($type))
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    inline bool isConstant() { return getConstant(); }

  private:
    static ::mlir::ParseResult parseGlobalInitialValue(::mlir::OpAsmParser &parser,
        ::mlir::Attribute &initialValue, ::mlir::TypeAttr typeAttr
    );

    static void printGlobalInitialValue(::mlir::OpAsmPrinter &printer, GlobalDefOp op,
        ::mlir::Attribute initialValue, ::mlir::TypeAttr typeAttr
    );
  }];
}

class GlobalRefOpBase<string mnemonic, list<Trait> traits = []>
    : GlobalDialectOp<
          mnemonic, traits#[DeclareOpInterfaceMethods<GlobalRefOpInterface>,
                            DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the definition for the `global` referenced in this op.
    inline ::mlir::FailureOr<SymbolLookupResult<GlobalDefOp>> getGlobalDefOp(::mlir::SymbolTableCollection &tables) {
      return ::llvm::cast<GlobalRefOpInterface>(getOperation()).getGlobalDefOp(tables);
    }
  }];
}

def LLZK_GlobalReadOp : GlobalRefOpBase<"read", [MemoryEffects<[MemRead]>]> {
  let summary = "read value of a global";
  let description = [{
    This operation reads the value of a named global.
  }];

  let arguments = (ins SymbolRefAttr:$name_ref);
  let results = (outs GlobalDefType:$val);

  let assemblyFormat = [{
    $name_ref `:` type($val) attr-dict
  }];
}

def LLZK_GlobalWriteOp
    : GlobalRefOpBase<"write", [WitnessGen, MemoryEffects<[MemWrite]>]> {
  let summary = "write value to a global";
  let description = [{
    This operation writes a value to a named global.
    Not allowed for globals declared with the "const" modifier.
  }];

  let arguments = (ins SymbolRefAttr:$name_ref, GlobalDefType:$val);

  let assemblyFormat = [{
    $name_ref `=` $val `:` type($val) attr-dict
  }];
}

#endif // LLZK_GLOBAL_OPS
