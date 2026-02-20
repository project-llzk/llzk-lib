//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FELT_OP_INTERFACES
#define LLZK_FELT_OP_INTERFACES

include "llzk/Dialect/Felt/IR/Types.td"

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

def FeltBinaryOpInterface : OpInterface<"FeltBinaryOpInterface"> {
  let description = [{
    Common interface for binary arithmetic operations on felts
  }];

  let cppNamespace = "::llzk::felt";

  let methods = [
      // Requires implementors to have '$lhs' and '$rhs' arguments.
      InterfaceMethod<[{Gets the left operand.}], "::mlir::Value", "getLhs",
                      (ins)>,
      InterfaceMethod<[{Gets the right operand.}], "::mlir::Value", "getRhs",
                      (ins)>];
}

#endif // LLZK_FELT_OP_INTERFACES
