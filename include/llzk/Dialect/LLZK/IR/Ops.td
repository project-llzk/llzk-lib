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
include "llzk/Dialect/LLZK/IR/Attrs.td" // must be included to generate docs
include "llzk/Dialect/Felt/IR/Types.td"
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
    : LLZKDialectOp<"nondet", [AlwaysSpeculatable,
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

    Note that `llzk.nondet` does not have the `ConstantLike` or `NoMemoryEffect`
    traits because different SSA variables initialized with `llzk.nondet` may be
    constrained in different ways so they cannot be treated as identical values.
    However, it does have the `AlwaysSpeculatable` trait to denote that it is
    free to move, hoist, and sick without changing the semantics.

    Example:

    ```llzk
    %0 = llzk.nondet : !felt.type
    %1 = llzk.nondet : !felt.type
    %c1 = felt.const 1
    constrain.eq %0, %c1 : !felt.type

    %m = struct.readm %self[@m] : !struct.type<@S>, !felt.type
    constrain.eq %m, %1 : !felt.type
    ```

    In the above example, `%m` is effectively unconstrained, as it is only constrained
    to the nondet value `%1`. However, if `%0` and `%1` were treated as pure constants,
    `%0` and `%1` could be combined, resulting in:

    ```llzk
    %0 = llzk.nondet : !felt.type
    %c1 = felt.const 1
    constrain.eq %0, %c1 : !felt.type

    %m = struct.readm %self[@m] : !struct.type<@S>, !felt.type
    constrain.eq %m, %0 : !felt.type
    ```

    This transformation would constrain `%m` to be exactly equal to 1, thus changing
    the semantics of the circuit.
  }];

  let results = (outs AnyLLZKType:$res);
  let assemblyFormat = [{ `:` type($res) attr-dict }];
  let hasCanonicalizer = 1;
}

def LLZK_AuxOp
    : LLZKDialectOp<"aux", [
          DeclareOpInterfaceMethods<
              OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "compiler-generated polynomial auxiliary";
  let description = [{
    This operation produces a compiler-generated existential SSA value for
    polynomial lowering.

    A polynomial auxiliary is placement-sensitive. Each dynamic execution
    instance of this operation denotes a fresh witness value in the region
    where the operation is placed. Lowering and materialization passes must not
    freely merge, hoist, sink, or otherwise move this operation across dynamic
    scope boundaries unless they have separately proved that the movement
    preserves the auxiliary's required freshness domain.

    This is intentionally distinct from `llzk.nondet`. User-level nondeterminism
    can be speculated, while generated polynomial auxiliaries carry semantic
    placement information that later materialization must respect. A later
    materializer may replace this operation with component storage, backend
    private signals, product-program values, or another representation only
    after proving that the chosen representation is at least as fresh as the
    dynamic domain required by the auxiliary's dependencies.

    V1 is conservative and only supports field elements.

    Example:

    ```llzk
    %aux = llzk.aux : !felt.type
    ```
  }];

  let results = (outs LLZK_FeltType:$res);
  let assemblyFormat = [{ `:` type($res) attr-dict }];
}

#endif // LLZK_OPS
