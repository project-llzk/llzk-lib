//===-- ZKLeanLeanOps.td -----------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANOPS_TD_
#define LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANOPS_TD_

include "ZKLeanLeanDialect.td"
include "ZKLeanLeanTypes.td"
include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class ZKLeanLeanOp<string mnemonic, list<Trait> traits = []>
    : Op<ZKLeanLean_Dialect, mnemonic, traits>;

//===------------------------------------------------------------------===//
// Struct definitions
//===------------------------------------------------------------------===//

def ZKLeanLean_StructDefOp
    : ZKLeanLeanOp<
          "structure", [HasParent<"::mlir::ModuleOp">, Symbol, SymbolTable,
                  IsolatedFromAbove,
                  NoRegionArguments]#GraphRegionNoTerminator.traits> {
  let summary = "ZKLean struct definition";
  let description = [{
    Defines a ZKLean struct with named members. Only member definitions are
    allowed in the body region.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{ $sym_name $bodyRegion attr-dict }];
}

def ZKLeanLean_MemberDefOp
    : ZKLeanLeanOp<"member", [HasParent<"::mlir::zkleanlean::StructDefOp">,
                               Symbol]> {
  let summary = "ZKLean struct member definition";
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<ZKExpr>:$type);
  let assemblyFormat = [{ $sym_name `:` $type attr-dict }];
}

//===------------------------------------------------------------------===//
// Struct operations
//===------------------------------------------------------------------===//

def ZKLeanLean_AccessorOp : ZKLeanLeanOp<"accessor", [Pure]> {
  let summary = "access member of a struct";
  let arguments = (ins ZKLeanLean_StructType:$component,
                       FlatSymbolRefAttr:$member_name);
  let results = (outs ZKExpr:$value);
  let assemblyFormat = [{
    $component `[` $member_name `]` `:` type($component) `,` type($value) attr-dict
  }];
}

//===------------------------------------------------------------------===//
// Lean operations
//===------------------------------------------------------------------===//

def ZKLeanLean_CallOp : ZKLeanLeanOp<"call", [Pure]> {
  let summary = "call a Lean function";
  let description = [{
    Represents a call to a Lean function by name with explicit argument and
    result types.
  }];

  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $args `)` `:` functional-type($args, results) attr-dict
  }];
}

#endif  // LIB_DIALECT_ZKLEANLEAN_ZKLEANLEANOPS_TD_
