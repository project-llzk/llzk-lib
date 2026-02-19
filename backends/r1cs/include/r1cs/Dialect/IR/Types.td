//===-- Types.td -----------------------------------------*--- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
#ifndef R1CS_TYPES
#define R1CS_TYPES

include "r1cs/Dialect/IR/Dialect.td"
include "mlir/IR/OpBase.td"

class R1CS_Type<string name, string storage> : TypeDef<R1CSDialect, name> {
  let mnemonic = storage;
}

def R1CS_SignalType : R1CS_Type<"Signal", "signal"> {
  let summary = "R1CS signal wire type";
}

def R1CS_LinearType : R1CS_Type<"Linear", "linear"> {
  let summary = "R1CS linear expression type";
}

#endif // R1CS_TYPES
