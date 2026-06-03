//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_OP_INTERFACES
#define LLZK_POD_OP_INTERFACES

include "llzk/Dialect/POD/IR/Types.td"

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PodRefOpInterface : OpInterface<"PodRefOpInterface"> {
  let description = [{
    Common interface for operations that reference a plain-old-data struct value.
  }];
  let cppNamespace = "::llzk::pod";

  let methods = [
      // Requires implementors to have a '$pod_ref' argument.
      InterfaceMethod<[{Gets the SSA Value for the referenced pod.}],
                      "::mlir::TypedValue<::llzk::pod::PodType>", "getPodRef",
                      (ins)>,
      InterfaceMethod<
          [{Gets the mutable operand slot holding the SSA Value for the referenced pod.}],
          "::mlir::OpOperand &", "getPodRefMutable", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Gets the type of the referenced pod.
    inline ::llzk::pod::PodType getPodRefType() { return getPodRef().getType(); }
  }];
}

def PodAccessOpInterface
    : OpInterface<"PodAccessOpInterface", [PodRefOpInterface]> {
  let description = [{
    Common interface for operations that read or write from a pod record.
  }];
  let cppNamespace = "::llzk::pod";

  let methods = [
      // Requires implementors to have a '$record_name' attribute.
      InterfaceMethod<
          [{Gets the record name attribute from the pod access op.}],
          "::mlir::FlatSymbolRefAttr", "getRecordNameAttr", (ins)>,
      InterfaceMethod<
          [{Return `true` if the op is a read, `false` if it's a write.}],
          "bool", "isRead", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Gets the record name as an attribute suitable for destructuring indices.
    inline ::mlir::StringAttr getRecordNameAsStringAttr() {
      return getRecordNameAttr().getLeafReference();
    }

    /// Required by companion interface DestructurableAccessorOpInterface / SROA pass
    bool canRewire(const ::mlir::DestructurableMemorySlot &slot,
            ::llvm::SmallPtrSetImpl<::mlir::Attribute> &usedIndices,
            ::mlir::SmallVectorImpl<::mlir::MemorySlot> &mustBeSafelyUsed,
            const ::mlir::DataLayout &dataLayout);

    /// Required by companion interface DestructurableAccessorOpInterface / SROA pass
    ::mlir::DeletionKind rewire(const ::mlir::DestructurableMemorySlot &slot,
            ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &subslots,
            ::mlir::OpBuilder &builder, const ::mlir::DataLayout &dataLayout);
  }];
}

#endif // LLZK_POD_OP_INTERFACES
