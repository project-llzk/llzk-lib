//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_OPS
#define LLZK_FELT_OPS

include "llzk/Dialect/Felt/IR/Dialect.td"
include "llzk/Dialect/Felt/IR/Types.td"
include "llzk/Dialect/Felt/IR/Attrs.td"
include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Shared/OpsBase.td"
include "llzk/Dialect/Felt/IR/OpInterfaces.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===------------------------------------------------------------------===//
// Op Classes
//===------------------------------------------------------------------===//

class FeltDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<FeltDialect, mnemonic, traits>;

class FeltDialectBinaryOp<string mnemonic, Type resultType,
                          list<Trait> traits = []>
    : BinaryOpBase<FeltDialect, mnemonic, resultType,
                   traits#[DeclareOpInterfaceMethods<FeltBinaryOpInterface>]>;

class FeltDialectUnaryOp<string mnemonic, Type resultType,
                         list<Trait> traits = []>
    : UnaryOpBase<FeltDialect, mnemonic, resultType, traits>;

//===------------------------------------------------------------------===//
// Constants
//===------------------------------------------------------------------===//

def LLZK_FeltConstantOp
    : FeltDialectOp<"const", [ConstantLike, Pure,
                              DeclareOpInterfaceMethods<
                                  OpAsmOpInterface, ["getAsmResultNames"]>,
                              TypeUnifyWithResult<"value">,
                              InferTypeOpAdaptorWithIsCompatible]> {
  let summary = "field element constant";
  let description = [{
    This operation produces a felt-typed SSA value holding an integer constant.

    Example:

    ```llzk
    %0 = felt.const 42
    %0 = felt.const 99 <"bn254">
    ```
  }];

  let arguments = (ins LLZK_FeltConstAttr:$value);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{ $value attr-dict }];
  let hasFolder = 1;
}

//===------------------------------------------------------------------===//
// Operators
//===------------------------------------------------------------------===//

def LLZK_AddFeltOp : FeltDialectBinaryOp<"add", LLZK_FeltType, [Commutative]> {
  let summary = "addition operator for field elements";
  let description = [{}];
}

def LLZK_SubFeltOp : FeltDialectBinaryOp<"sub", LLZK_FeltType> {
  let summary = "subtraction operator for field elements";
  let description = [{}];
}

def LLZK_MulFeltOp : FeltDialectBinaryOp<"mul", LLZK_FeltType, [Commutative]> {
  let summary = "multiplication operator for field elements";
  let description = [{}];
}

def LLZK_PowFeltOp
    : FeltDialectBinaryOp<"pow", LLZK_FeltType, [NotFieldNative]> {
  let summary = "exponentiation operator for field elements";
  let description = [{

  Raises a field element to the power of an exponent.

  ```llzk
  %result = felt.pow %base, %exponent
  ```

  }];
}

def LLZK_DivFeltOp : FeltDialectBinaryOp<"div", LLZK_FeltType> {
  let summary = "division operator for field elements";
  let description = [{}];
}

def LLZK_UnsignedIntDivFeltOp
    : FeltDialectBinaryOp<"uintdiv", LLZK_FeltType, [NotFieldNative]> {
  let summary = "unsigned integer division operator for field elements";
  let description = [{
    Treats the operands as if they were unsigned integers with bitwidth
    equal to that of the prime modulus and performs division rounding towards zero.
  }];
}

def LLZK_SignedIntDivFeltOp
    : FeltDialectBinaryOp<"sintdiv", LLZK_FeltType, [NotFieldNative]> {
  let summary = "signed integer division operator for field elements";
  let description = [{
    Treats the operands as if they were signed integers with bitwidth
    equal to that of the prime modulus (no additional sign bit is added)
    and performs division rounding towards zero.

    The signed integer representation of felt `f` in prime field with modulus
    `p` follows the following formula:

        signed_int(f) = f    if 0 <= f < p/2 + 1
            "p/2" here is unsigned integer division rounding towards 0
        signed_int(f) = f-p  if p/2 + 1 <= f < p
  }];
}

def LLZK_UnsignedModFeltOp
    : FeltDialectBinaryOp<"umod", LLZK_FeltType, [NotFieldNative]> {
  let summary =
      "unsigned integer modulus/remainder operator for field elements";
  let description = [{
    Computes the remainder that would result from the division operation performed
    by `felt.uintdiv`.
  }];
}

def LLZK_SignedModFeltOp
    : FeltDialectBinaryOp<"smod", LLZK_FeltType, [NotFieldNative]> {
  let summary = "signed integer modulus/remainder operator for field elements";
  let description = [{
    Computes the remainder that would result from the division operation performed
    by `felt.sintdiv`.
  }];
}

def LLZK_NegFeltOp : FeltDialectUnaryOp<"neg", LLZK_FeltType> {
  let summary = "negation operator for field elements";
  let description = [{}];
}

def LLZK_InvFeltOp
    : FeltDialectUnaryOp<"inv", LLZK_FeltType, [NotFieldNative]> {
  let summary = "inverse operator for field elements";
  let description = [{}];
}

def LLZK_AndFeltOp
    : FeltDialectBinaryOp<"bit_and",
                          LLZK_FeltType, [NotFieldNative, Commutative]> {
  let summary = "bitwise AND operator for field elements";
  let description = [{}];
}

def LLZK_OrFeltOp
    : FeltDialectBinaryOp<"bit_or",
                          LLZK_FeltType, [NotFieldNative, Commutative]> {
  let summary = "bitwise OR operator for field elements";
  let description = [{}];
}

def LLZK_XorFeltOp
    : FeltDialectBinaryOp<"bit_xor",
                          LLZK_FeltType, [NotFieldNative, Commutative]> {
  let summary = "bitwise XOR operator for field elements";
  let description = [{}];
}

def LLZK_NotFeltOp
    : FeltDialectUnaryOp<"bit_not", LLZK_FeltType, [NotFieldNative]> {
  let summary = "integer complement (bitwise-not) operator for field elements";
  let description = [{
    Treats the operand as an integer with a bitwidth equal to the bitwidth
    of the prime field's modulus and compute the one's complement of the integer.
    The result is converted back to a field element by applying the prime modulus.
  }];
}

def LLZK_ShlFeltOp
    : FeltDialectBinaryOp<"shl", LLZK_FeltType, [NotFieldNative]> {
  let summary = "left shift operator for field elements";
  let description = [{}];
}

def LLZK_ShrFeltOp
    : FeltDialectBinaryOp<"shr", LLZK_FeltType, [NotFieldNative]> {
  let summary = "right shift operator for field elements";
  let description = [{}];
}

#endif // LLZK_FELT_OPS
