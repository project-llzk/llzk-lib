//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_RAM_OPS
#define LLZK_RAM_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

include "llzk/Dialect/Felt/IR/Types.td"
include "llzk/Dialect/RAM/IR/Dialect.td"
include "llzk/Dialect/Function/IR/OpTraits.td"

class RAMDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<RAMDialect, mnemonic, traits>;

def LLZK_LoadOp : RAMDialectOp<"load", [WitnessGen, MemoryEffects<[MemRead]>]> {
  let summary = "read a value from memory";
  let description = [{
    Reads the felt stored at cell `%addr` of the flat, cell-addressed RAM.

    `%addr` is a slot index. Each cell holds exactly one
    `!felt.type` value; producers storing other types (booleans, integers,
    composites) must widen or lay them out across cells themselves.

    Example:

    ```llzk
    %val = ram.load %addr : !felt.type
    ```
  }];

  let arguments = (ins Index:$addr);
  let results = (outs LLZK_FeltType:$val);

  let assemblyFormat = [{
    $addr `:` type($val) attr-dict
  }];
}

def LLZK_StoreOp
    : RAMDialectOp<"store", [WitnessGen, MemoryEffects<[MemWrite]>]> {
  let summary = "write a value to memory";
  let description = [{
    Writes felt `%val` to cell `%addr` of the flat, cell-addressed RAM.

    `%addr` is a slot index. Each cell holds exactly one
    `!felt.type` value; producers storing other types (booleans, integers,
    composites) must widen or lay them out across cells themselves.

    Example:

    ```llzk
    ram.store %addr, %val : !felt.type
    ```
  }];

  let arguments = (ins Index:$addr, LLZK_FeltType:$val);

  let assemblyFormat = [{
    $addr `,` $val `:` type($val) attr-dict
  }];
}

#endif // LLZK_RAM_OPS
