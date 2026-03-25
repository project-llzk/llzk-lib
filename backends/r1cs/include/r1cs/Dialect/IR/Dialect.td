//===-- Dialect.td -----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef R1CS_DIALECT
#define R1CS_DIALECT

include "mlir/IR/DialectBase.td"

def R1CSDialect : Dialect {
  let name = "r1cs";
  let summary = "R1CS";
  let cppNamespace = "::r1cs";
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = true;
  let useDefaultAttributePrinterParser = true;
}

#endif
