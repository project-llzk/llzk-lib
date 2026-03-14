//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POLYMORPHIC_OP_INTERFACES
#define LLZK_POLYMORPHIC_OP_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/IR/SymbolInterfaces.td"

def ConstParamSymbolOpInterface
    : OpInterface<"ConstParamSymbolOpInterface", [Symbol]> {
  let description = [{
    Common interface for constant parameter-like symbol operations that can be
    defined within a `TemplateOp`
  }];

  let cppNamespace = "::llzk::polymorphic";
}

#endif // LLZK_POLYMORPHIC_OP_INTERFACES
