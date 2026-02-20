//===-- Ops.td -------------------------------------------*--- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef R1CS_OPS
#define R1CS_OPS

include "r1cs/Dialect/IR/Types.td"
include "r1cs/Dialect/IR/Attrs.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class R1CS_Op<string mnemonic, list<Trait> traits = []>
    : Op<R1CSDialect, mnemonic, traits>;

def R1CS_CircuitDefOp : R1CS_Op<"circuit", [Symbol, IsolatedFromAbove,
                                            SingleBlock, NoTerminator]> {
  let summary = "Defines an R1CS circuit with labeled inputs and constraints";
  let description = [{
    The `r1cs.circuit` operation defines a named circuit, with explicit inputs
    and internally declared signals and constraints.

    Signals defined with `r1cs.def` and marked `public` are considered outputs.

    Example:

    ```mlir
    r1cs.circuit @CmpConstraint inputs (%arg0: !r1cs.signal {#r1cs.pub}, %arg1: !r1cs.signal, %arg2: !r1cs.signal) {
        %0 = r1cs.def 0 : !r1cs.signal {pub = #r1cs.pub}
        %1 = r1cs.def 1 : !r1cs.signal
        %2 = r1cs.def 2 : !r1cs.signal
        %3 = r1cs.to_linear %arg1 : !r1cs.signal to !r1cs.linear
        %4 = r1cs.to_linear %arg2 : !r1cs.signal to !r1cs.linear
        %5 = r1cs.to_linear %1 : !r1cs.signal to !r1cs.linear
        %6 = r1cs.mul_const %5,  -1 : i64 : !r1cs.linear
        r1cs.constrain %3, %4, %6 : !r1cs.linear
    }
    ```
  }];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<DictionaryAttr>:$arg_attrs);

  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>, ];

  let extraClassDeclaration = [{
    ::mlir::Block* addEntryBlock();
  }];
}

def R1CS_SignalDefOp : R1CS_Op<"def", [Pure]> {
  let summary = "Define a signal with label and optional public visibility";
  let description = [{
    The `def` op creates a new signal associated with a numeric label.

    If the optional `public` keyword is present, the signal is treated as a public output
    of the circuit. Otherwise, it is considered private/internal.

    Example:
    ```
      %0 = r1cs.def 0 : !r1cs.signal {pub = #r1cs.public} // public signal
      %1 = r1cs.def 1 : !r1cs.signal // private
    ```
  }];
  let arguments = (ins UI32Attr:$label, OptionalAttr<PublicAttr>:$pub);
  let results = (outs R1CS_SignalType:$out);
  let assemblyFormat = [{
    $label `:` type($out) attr-dict
  }];
}

def R1CS_ToLinearOp : R1CS_Op<"to_linear", [Pure]> {
  let summary = "Casts a signal to a linear expression";
  let arguments = (ins R1CS_SignalType:$input);
  let results = (outs R1CS_LinearType:$output);
  let assemblyFormat = "$input `:` type($input) `to` type($output) attr-dict";
}

def R1CS_ConstOp : R1CS_Op<"const", [Pure]> {
  let summary = "Create a constant linear value";
  let arguments = (ins FeltAttr:$value);
  let results = (outs R1CS_LinearType:$out);
  let assemblyFormat = "$value attr-dict `:` type($out)";
}

def R1CS_AddOp : R1CS_Op<"add", [Pure, Commutative]> {
  let summary = "Add two linear expressions";
  let arguments = (ins R1CS_LinearType:$lhs, R1CS_LinearType:$rhs);
  let results = (outs R1CS_LinearType:$sum);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($sum)";
}

def R1CS_MulConstOp : R1CS_Op<"mul_const", [Pure, Commutative]> {
  let summary = "Scale a linear expression by a constant";
  let arguments = (ins R1CS_LinearType:$input, FeltAttr:$constValue);
  let results = (outs R1CS_LinearType:$scaled);
  let assemblyFormat = "$input `,` $constValue attr-dict `:` type($scaled)";
}

def R1CS_NegOp : R1CS_Op<"neg", [Pure]> {
  let summary = "Negate a linear expression";
  let arguments = (ins R1CS_LinearType:$input);
  let results = (outs R1CS_LinearType:$neg);
  let assemblyFormat = "$input attr-dict `:` type($neg)";
}

def R1CS_ConstrainOp : R1CS_Op<"constrain", []> {
  let summary = "Enforce a * b - c = 0 R1CS constraint";
  let arguments = (ins R1CS_LinearType:$a, R1CS_LinearType:$b,
      R1CS_LinearType:$c);
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a)";
}

#endif // R1CS_OPS
