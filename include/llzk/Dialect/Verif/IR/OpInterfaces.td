//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_VERIF_OP_INTERFACES
#define LLZK_VERIF_OP_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

def ConditionOpInterface
    : OpInterface<"ConditionOpInterface", [DeclareOpInterfaceMethods<
                                              MemoryEffectsOpInterface>]> {
  let description = [{
    Common interface for precondition and postcondition operations.

    This declares the `MemoryEffectsOpInterface`, which, like the `cf.assert` (MLIR `cf` dialect)
    and `bool.assert` (LLZK `bool` dialect) ops, adds a MemWrite affect to model program termination.
  }];
  let cppNamespace = "::llzk::verif";

  let methods = [
      // Requires implementors to have a condition argument
      InterfaceMethod<[{Gets the SSA Value for the condition operand.}],
                      "::mlir::TypedValue<::mlir::IntegerType>", "getCondition",
                      (ins)>];
}

def PreconditionOpInterface
    : OpInterface<"PreconditionOpInterface", [ConditionOpInterface]> {
  let description = [{
    Common interface for precondition operations (i.e., `require_*`).
  }];
  let cppNamespace = "::llzk::verif";
}

def PostconditionOpInterface
    : OpInterface<"PostconditionOpInterface", [ConditionOpInterface]> {
  let description = [{
    Common interface for postcondition operations (i.e., `ensure_*`).
  }];
  let cppNamespace = "::llzk::verif";
}

#endif // LLZK_VERIF_OP_INTERFACES
