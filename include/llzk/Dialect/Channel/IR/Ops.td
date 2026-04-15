//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CHANNEL_OPS
#define LLZK_CHANNEL_OPS

include "llzk/Dialect/Channel/IR/Dialect.td"
include "llzk/Dialect/Felt/IR/Types.td"
include "llzk/Dialect/Function/IR/OpTraits.td"
include "llzk/Dialect/Shared/Types.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class ChannelDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<ChannelDialect, mnemonic, traits>;

def ChannelDefOp : ChannelDialectOp<"def", [HasParent<"mlir::ModuleOp">, Symbol]> {
  let summary = "channel declaration";
  let description = [{
    Declares a channel symbol for the channel API used to model multiset
    equality constraints across interacting LLZK structs. A channel represents
    an unordered multiset of row-local tuples rather than an ordered stream.
    The payload signature is declared explicitly on the channel definition. In
    this task, the signature is a builtin tuple type whose elements are all
    `!felt.type`. Multiplicity is not part of the declaration; it remains a
    per-use operand on `channel.push` and `channel.pull`.

    Example:

    ```llzk
    channel.def @c : tuple<!felt.type, !felt.type>
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$message_type);

  let assemblyFormat = [{ $sym_name `:` $message_type attr-dict }];
  let hasVerifier = 1;
}

class ChannelUseOpBase<string mnemonic, list<Trait> traits = []>
    : ChannelDialectOp<
          mnemonic, traits#[AttrSizedOperandSegments, ConstraintGen,
                            DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let arguments = (ins FlatSymbolRefAttr:$channel_ref,
      Variadic<AnyLLZKType>:$tuple, Optional<AnyLLZKType>:$mult);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def ChannelPushOp : ChannelUseOpBase<"push"> {
  let summary = "push tuples into a channel";
  let description = [{
    Adds one or more row-local values as a tuple contribution to a channel
    multiset. The optional `mult` operand specifies the tuple multiplicity; if
    omitted, it defaults to `1`.

    Examples:

    ```llzk
    channel.push @c (%x, %y) : !felt.type, !felt.type
    channel.push @c (%x, %y) { mult = %m } : !felt.type, !felt.type, !felt.type
    ```
  }];
}

def ChannelPullOp : ChannelUseOpBase<"pull"> {
  let summary = "pull tuples from a channel";
  let description = [{
    Removes one or more row-local values as a tuple contribution from a
    channel multiset. The optional `mult` operand specifies the tuple
    multiplicity; if omitted, it defaults to `1`.

    Examples:

    ```llzk
    channel.pull @c (%x, %y) : !felt.type, !felt.type
    channel.pull @c (%x, %y) { mult = %m } : !felt.type, !felt.type, !felt.type
    ```
  }];
}

#endif // LLZK_CHANNEL_OPS
