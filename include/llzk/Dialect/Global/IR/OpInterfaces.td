//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_GLOBAL_OP_INTERFACES
#define LLZK_GLOBAL_OP_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def GlobalMemoryResource : Resource<"::llzk::global::GlobalMemoryResource">;

def GlobalRefOpInterface : OpInterface<"GlobalRefOpInterface"> {
  let description = [{
    Common interface for operations that read or write from a global.
  }];
  let cppNamespace = "::llzk::global";

  let methods = [
      // Requires implementors to have a '$val' argument.
      InterfaceMethod<
          [{Gets the SSA Value that holds the read/write data for the GlobalRefOp.}],
          "::mlir::Value", "getVal", (ins)>,
      // Requires implementors to have a '$name_ref' argument.
      InterfaceMethod<[{Gets the global name attribute from the GlobalRefOp.}],
                      "::mlir::SymbolRefAttr", "getNameRef", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Gets the definition for the `global` referenced in this op.
    ::mlir::FailureOr<SymbolLookupResult<GlobalDefOp>> getGlobalDefOp(::mlir::SymbolTableCollection &tables);
  }];
}

#endif // LLZK_GLOBAL_OP_INTERFACES
