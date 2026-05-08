//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_VERIF_OPS
#define LLZK_VERIF_OPS

include "llzk/Dialect/Verif/IR/Dialect.td"
include "llzk/Dialect/Verif/IR/OpInterfaces.td"
include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class VerifDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<VerifDialect, mnemonic, traits>;

class ConditionOp<string mnemonic, list<Trait> traits = []>
    : VerifDialectOp<mnemonic,
                     traits#[ConditionOpInterface,
                             HasAncestor<"::llzk::verif::ContractOp">]> {

  let arguments = (ins I1:$condition);
  let results = (outs);

  let extraClassDefinition = [{
    // This side effect models "program termination". Based on
    // https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
    void $cppClass::getEffects(
        ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects
    ) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get());
    }
  }];
}

//===------------------------------------------------------------------===//
// Precondition operations
//===------------------------------------------------------------------===//

class RequireOpBase<string mnemonic, list<Trait> traits = []>
    : ConditionOp<mnemonic, traits#[PreconditionOpInterface]>;

def RequireComputeOp : RequireOpBase<"require_compute"> {
  let summary = "";
  let description = [{
  }];
}

def RequireConstrainOp : RequireOpBase<"require_constrain"> {
  let summary = "";
  let description = [{
  }];
}

//===------------------------------------------------------------------===//
// Postcondition operations
//===------------------------------------------------------------------===//

class EnsureOpBase<string mnemonic, list<Trait> traits = []>
    : ConditionOp<mnemonic, traits#[PostconditionOpInterface]>;

def EnsureComputeOp : EnsureOpBase<"ensure_compute"> {
  let summary = "";
  let description = [{
  }];
}

def EnsureConstrainOp : EnsureOpBase<"ensure_constrain"> {
  let summary = "";
  let description = [{
  }];
}

//===------------------------------------------------------------------===//
// ContractDefOp
//===------------------------------------------------------------------===//

def ContractOp
    : VerifDialectOp<
          "contract", [ParentOneOf<["::mlir::ModuleOp",
                                    "::llzk::polymorphic::TemplateOp"]>,
                       DeclareOpInterfaceMethods<SymbolUserOpInterface>, Symbol,
                       LLZKSymbolTable, IsolatedFromAbove]> {
  let summary = "defines a specification contract for a given symbol";
  let description = [{
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      SymbolNameAttr:$target_sym_name, TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);

  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // LLZK_VERIF_OPS
