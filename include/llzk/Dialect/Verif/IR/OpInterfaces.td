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
include "mlir/IR/SymbolInterfaces.td"

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

def InvariantTarget : OpInterface<"InvariantTargetOpInterface"> {
  let description = [{
    Interface implemented by operations that can be targeted by a loop invariant.

    Targets are referenced by a label that implementations of this interface must provide.
  }];
  let cppNamespace = "::llzk::verif";

  let methods = [InterfaceMethod<[{Gets the label of the loop, if available.}],
                                 "::mlir::FailureOr<::mlir::StringRef>",
                                 "getLabel", (ins), [{
                                   auto attr = ::mlir::dyn_cast_or_null<::mlir::StringAttr>($_op->getDiscardableAttr("loop_label"));
                                   if (!attr) return ::mlir::failure();
                                   return attr.getValue();
                                 }]>,
                 InterfaceMethod<
                     [{Gets the types of the values that the invariant binds inside its body.}],
                     "::mlir::SmallVector<::mlir::Type>", "getArgumentTypes",
                     (ins)>,
  ];
}

def ContractTarget
    : OpInterface<"ContractTargetOpInterface", [Symbol]> {
  let description = [{
    Interface implemented by operations that can be targeted by a contract.

    Targets are referenced by name so implementations of this interface must also implement the `Symbol` interface.
  }];
  let cppNamespace = "::llzk::verif";

  let methods = [InterfaceMethod<
      [{Gets the ops that can be targeted by invariant ops.}],
      "::llvm::SmallVector<::llzk::verif::InvariantTargetOpInterface>",
      "getLoops", (ins), [{
                        ::mlir::SmallVector<::llzk::verif::InvariantTargetOpInterface> targets;
                        $_op->walk([&targets](::llzk::verif::InvariantTargetOpInterface target) {
                            targets.push_back(target);
                          });
                        return targets;
                      }]>];

  let extraClassDeclaration = [{
    /// Emulates the static method with the same name found in operations.
    static ::mlir::StringLiteral getOperationName() {
      return ::mlir::StringLiteral("contract target interface implementation");
    }                    
  }];
}

#endif // LLZK_VERIF_OP_INTERFACES
