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

    An initializer's typed attributes must exactly match the declared global
    type. When parsing a felt initializer, an explicitly selected field
    refines an unspecified declared felt type; conversely, an unspecified
    felt initializer adopts an explicitly declared field. Thus, an initialized
    global always stores matching declared and initializer felt types.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, UnitAttr:$constant,
      TypeAttrOf<GlobalDefType>:$type,
      DefaultValuedAttr<AnyAttr, "nullptr">:$initial_value);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns true when this def is marked as `const`.
    inline bool isConstant() { return getConstant(); }
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

def LLZK_GlobalReadOp
    : GlobalRefOpBase<
          "read", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "read value of a global";
  let description = [{
    This operation reads the value of a named global. Reads marked `const` must target
    a global declared with the `const` modifier and thus have no memory effects.
  }];

  let arguments = (ins UnitAttr:$constant, SymbolRefAttr:$name_ref);
  let results = (outs GlobalDefType:$val);

  let assemblyFormat = [{
    (`const` $constant^)? $name_ref `:` type($val) attr-dict
  }];

  // `constant` was added in version 3. Preserve the v2 bytecode layout, which
  // contains only `name_ref`, when decoding older bytecode.
  let useCustomPropertiesEncoding = 1;

  let extraClassDeclaration = [{
    /// Returns true when this read is marked as `const`.
    inline bool isConstant() { return getConstant(); }
  }];
}

def LLZK_GlobalWriteOp
    : GlobalRefOpBase<
          "write", [WitnessGen,
                    MemoryEffects<[MemWrite<GlobalMemoryResource>]>]> {
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
