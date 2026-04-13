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

def LLZK_AllocOp : RAMDialectOp<"alloc", [WitnessGen]> {
  let summary = "allocate a memory region";
  let description = [{
    Allocates a memory region of the given size and returns a handle.

    Example:

    ```llzk
    %mem = ram.alloc %size : index
    ```
  }];

  let arguments = (ins Index:$size);
  let results = (outs Index:$handle);

  let assemblyFormat = [{
    $size `:` type($size) attr-dict
  }];
}

def LLZK_LoadOp : RAMDialectOp<"load", [WitnessGen]> {
  let summary = "read a value from memory";
  let description = [{
    Reads a value from the memory region at the given address.

    Example:

    ```llzk
    %val = ram.load %mem[%addr] : index, !felt.type
    ```
  }];

  let arguments = (ins Index:$mem, Index:$addr);
  let results = (outs AnyLLZKType:$val);

  let assemblyFormat = [{
    $mem `[` $addr `]` `:` type($mem) `,` type($val) attr-dict
  }];
}

def LLZK_StoreOp : RAMDialectOp<"store", [WitnessGen]> {
  let summary = "write a value to memory";
  let description = [{
    Writes a value to the memory region at the given address.

    Example:

    ```llzk
    ram.store %mem[%addr] = %val : index, !felt.type
    ```
  }];

  let arguments = (ins Index:$mem, Index:$addr, AnyLLZKType:$val);

  let assemblyFormat = [{
    $mem `[` $addr `]` `=` $val `:` type($mem) `,` type($val) attr-dict
  }];
}

#endif // LLZK_RAM_OPS
