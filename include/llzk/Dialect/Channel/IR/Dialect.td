//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_CHANNEL_DIALECT
#define LLZK_CHANNEL_DIALECT

include "mlir/IR/DialectBase.td"

def ChannelDialect : Dialect {
  let name = "channel";
  let summary = "LLZK multiset-equality operations for row-local cross-struct "
                "interaction.";
  let description = [{
    The `channel` dialect provides the IR surface for expressing multiset
    equality constraints between interacting LLZK structs. A channel
    models an unordered multiset of row-local tuples rather than an ordered
    stream, making it suitable for lookup-style and producer/consumer
    consistency constraints without introducing sequencing semantics.

    Channel interactions are represented by `channel.push` and `channel.pull`,
    which contribute tuples and optional multiplicities to the multiset. Tuple
    elements are expected to be algebraic expressions over a single row
    context. Channel declarations carry an explicit payload signature; in this
    task that signature is a tuple of `!felt.type` elements. Multiplicity is
    per-use metadata and is not part of the declaration.

    This dialect is responsible for the channel-specific IR structure and
    verification rules. The backend is responsible for enforcing the global
    equality condition between the pushed and pulled multisets.
  }];

  let cppNamespace = "::llzk::channel";
}

#endif // LLZK_CHANNEL_DIALECT
