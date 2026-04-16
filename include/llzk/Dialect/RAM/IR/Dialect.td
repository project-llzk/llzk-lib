//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_RAM_DIALECT
#define LLZK_RAM_DIALECT

include "mlir/IR/DialectBase.td"

def RAMDialect : Dialect {
  let name = "ram";
  let summary = "LLZK support for random access memory during witness generation.";

  let cppNamespace = "::llzk::ram";
}

#endif // LLZK_RAM_DIALECT
