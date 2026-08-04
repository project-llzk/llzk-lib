//===-- Ops.td -------------------------------------------*--- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef PCL_OPS
#define PCL_OPS

include "Types.td"
include "Attrs.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def InsideFunc
    : PredOpTrait<"must be inside a func.func",
                  CPred<"$_op.getParentOfType<mlir::func::FuncOp>()">>;

class PCL_Op<string mnemonic, list<Trait> traits = []>
    : Op<PCLDialect, mnemonic, traits>;

// -----------------------------------------------------------------------------
// Variable declaration
// -----------------------------------------------------------------------------

def PCL_VarOp : PCL_Op<"var", [Pure, InsideFunc]> {
  let summary = "Declare a felt variable.";
  let description = [{
    The `var` op creates a new variable associated with a name. It additionally takes a parameter
    `is_output` which indicates an output variable if true and an intermediate otherwise.

    Example:
    ```
      %0 = pcl.def "foo" true // output
      %1 = pcl.def "bar" false // intermediate
    ```
  }];
  let arguments = (ins StrAttr:$name, BoolAttr:$is_output);
  let results = (outs PCL_FieldType:$res);
  let assemblyFormat = "$name $is_output attr-dict";
}

// -----------------------------------------------------------------------------
// Helper classes
// -----------------------------------------------------------------------------

class PCL_BinOp<string mnemonic, Type opType, Type resType = opType,
                list<Trait> traits = []>
    : PCL_Op<mnemonic,
             traits#[Pure, InsideFunc]#!if(
                 !eq(opType, resType), [SameOperandsAndResultType], [])> {
  let arguments = (ins opType:$lhs, opType:$rhs);
  let results = (outs resType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

class PCL_FieldBinOp<string mnemonic, list<Trait> traits = []>
    : PCL_BinOp<mnemonic, PCL_FieldType, PCL_FieldType, traits>;

class PCL_BoolBinOp<string mnemonic, list<Trait> traits = []>
    : PCL_BinOp<mnemonic, PCL_BoolType, PCL_BoolType, traits>;

class PCL_UnOp<string mnemonic, Type opType, Type resType = opType,
               list<Trait> traits = []>
    : PCL_Op<mnemonic,
             traits#[Pure, InsideFunc]#!if(
                 !eq(opType, resType), [SameOperandsAndResultType], [])> {
  let arguments = (ins opType:$value);
  let results = (outs resType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

// -----------------------------------------------------------------------------
// Arithmetic over fields
// -----------------------------------------------------------------------------

def PCL_ConstOp : PCL_Op<"const", [Pure, InsideFunc, ConstantLike]> {
  let summary = "Constant felt";
  let arguments = (ins FeltAttr:$value);
  let results = (outs PCL_FieldType:$res);
  let assemblyFormat = "$value attr-dict";
  let extraClassDeclaration = [{
    auto getValueAPInt() -> ::llvm::APInt { return getValue().getValue(); }
  }];
  let hasFolder = 1;
}

def PCL_AddOp : PCL_FieldBinOp<"add", [Commutative]> {
  let summary = "Field addition";
}

def PCL_SubOp : PCL_FieldBinOp<"sub"> {
  let summary = "Field subtraction";
  let hasCanonicalizer = 1;
}

def PCL_MulOp : PCL_FieldBinOp<"mul", [Commutative]> {
  let summary = "Field multiplication";
  let hasCanonicalizer = 1;
}

def PCL_NegOp : PCL_UnOp<"neg", PCL_FieldType> {
  let summary = "Field negation";
  let hasCanonicalizer = 1;
}

// -----------------------------------------------------------------------------
// Conversion from bool to felt.
// -----------------------------------------------------------------------------

def PCL_AsFeltOp : PCL_UnOp<"asfelt", PCL_BoolType, PCL_FieldType> {
  let summary = "Bool to felt conversion";
}

// -----------------------------------------------------------------------------
// Comparisons: return !pcl.bool
// -----------------------------------------------------------------------------

class PCLCmpBase<string mnem, list<Trait> traits = []>
    : PCL_BinOp<mnem, PCL_FieldType, PCL_BoolType, traits>;

def PCL_CmpEqOp : PCLCmpBase<"eq", [Commutative]> {
  let summary = "lhs == rhs (field equality)";
  let hasCanonicalizer = 1;
}
def PCL_CmpLtOp : PCLCmpBase<"lt"> { let summary = "lhs < rhs"; }
def PCL_CmpLeOp : PCLCmpBase<"le"> { let summary = "lhs <= rhs"; }
def PCL_CmpGtOp : PCLCmpBase<"gt"> { let summary = "lhs > rhs"; }
def PCL_CmpGeOp : PCLCmpBase<"ge"> { let summary = "lhs >= rhs"; }

// -----------------------------------------------------------------------------
// Boolean constants
// -----------------------------------------------------------------------------

class PCL_ConstBoolOp<string mnemonic>
    : PCL_Op<mnemonic, [Pure, InsideFunc, ConstantLike]> {
  let summary = "constant "#mnemonic#" value";
  let results = (outs PCL_BoolType:$res);
  let assemblyFormat = "attr-dict";
  let hasFolder = 1;
}

def PCL_TrueOp : PCL_ConstBoolOp<"true">;
def PCL_FalseOp : PCL_ConstBoolOp<"false">;

// -----------------------------------------------------------------------------
// Boolean connectives on !pcl.bool
// -----------------------------------------------------------------------------

def PCL_AndOp : PCL_BoolBinOp<"and", [Commutative]> {
  let summary = "Boolean conjunction (lhs && rhs)";
}

def PCL_OrOp : PCL_BoolBinOp<"or", [Commutative]> {
  let summary = "Boolean disjunction (lhs || rhs)";
}

def PCL_NotOp : PCL_UnOp<"not", PCL_BoolType> {
  let summary = "Boolean negation (!cond)";
  let hasCanonicalizer = 1;
}

def PCL_ImpliesOp : PCL_BoolBinOp<"implies"> {
  let summary = "Boolean implication (lhs => rhs)";
}

def PCL_IffOp : PCL_BoolBinOp<"iff", [Commutative]> {
  let summary = "Boolean equivalence (lhs <=> rhs)";
}

def PCL_DetOp : PCL_UnOp<"det", PCL_FieldType, PCL_BoolType> {
  let summary = "returns true iff $expr deterministic";
}

// -----------------------------------------------------------------------------
// Assertions & post-conditions (side-effecting)
// -----------------------------------------------------------------------------

def PCL_AssertOp : PCL_Op<"assert", []> {
  let summary = "Assert that a boolean holds";
  let arguments = (ins PCL_BoolType:$cond);
  let assemblyFormat = "$cond attr-dict";
  let hasCanonicalizer = 1;
}

def PCL_PostOp : PCL_Op<"post_cond", []> {
  let summary = "Post-condition on function outputs";
  let arguments = (ins PCL_BoolType:$cond);
  let assemblyFormat = "$cond attr-dict";
  let hasCanonicalizer = 1;
}

def PCL_AssumeDeterministicOp : PCL_Op<"assume.deterministic", []> {
  let summary = "Assume the given value is deterministic";
  let arguments = (ins PCL_FieldType:$v);
  let assemblyFormat = "$v attr-dict";
}

#endif // PCL_OPS
