//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_OPS
#define LLZK_OPS

include "llzk/Dialect/LLZK/IR/Dialect.td"
include "llzk/Dialect/Shared/Types.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===------------------------------------------------------------------===//
// Op Classes
//===------------------------------------------------------------------===//

class LLZKDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<LLZKDialect, mnemonic, traits>;

def LLZK_NonDetOp
    : LLZKDialectOp<"nondet", [ConstantLike, Pure,
                               DeclareOpInterfaceMethods<
                                   OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "uninitialized variable";
  let description = [{
    This operation produces an SSA variable of the specified type but with
    nondeterministic value. 

    This op can be used in `@constrain()` functions in place of expressions that
    cannot be included in constraints. It may also be generated for a frontend
    language that supports uninitialized variables and can also be introduced by
    the `llzk-array-to-scalar` pass if there is a read from an array index
    that was not dominated by an earlier write to that same index.

    Example:

    ```llzk
    %0 = llzk.nondet : !felt.type
    ```
  }];

  let results = (outs AnyLLZKType:$res);
  let assemblyFormat = [{ `:` type($res) attr-dict }];
}

#endif // LLZK_OPS
