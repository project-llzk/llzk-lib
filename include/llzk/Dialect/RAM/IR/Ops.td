//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_RAM_OPS
#define LLZK_RAM_OPS

include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/RAM/IR/Dialect.td"
include "llzk/Dialect/Function/IR/OpTraits.td"

class RAMDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<RAMDialect, mnemonic, traits>;

def LLZK_LoadOp : RAMDialectOp<"load", [WitnessGen]> {
  let summary = "read a value from memory";
  let description = [{
    Reads a value from the flat memory region at the given address.

    Example:

    ```llzk
    %val = ram.load %addr : !felt.type
    ```
  }];

  let arguments = (ins Index:$addr);
  let results = (outs AnyLLZKType:$val);

  let assemblyFormat = [{
    $addr `:` type($val) attr-dict
  }];
}

def LLZK_StoreOp : RAMDialectOp<"store", [WitnessGen]> {
  let summary = "write a value to memory";
  let description = [{
    Writes a value to the flat memory region at the given address.

    Example:

    ```llzk
    ram.store %addr, %val : !felt.type
    ```
  }];

  let arguments = (ins Index:$addr, AnyLLZKType:$val);

  let assemblyFormat = [{
    $addr `,` $val `:` type($val) attr-dict
  }];
}

#endif // LLZK_RAM_OPS
