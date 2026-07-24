//===-- OpInterfaces.td ------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ARRAY_OP_INTERFACES
#define LLZK_ARRAY_OP_INTERFACES

include "llzk/Dialect/Array/IR/Types.td"

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"

def ArrayRefOpInterface : OpInterface<"ArrayRefOpInterface"> {
  let description = [{
    Common interface for operations that reference an array value.
  }];
  let cppNamespace = "::llzk::array";

  let methods = [
      // Requires implementors to have a '$arr_ref' argument.
      InterfaceMethod<[{Gets the SSA Value for the referenced array.}],
                      "::mlir::TypedValue<::llzk::array::ArrayType>",
                      "getArrRef", (ins)>,
      InterfaceMethod<
          [{Gets the mutable operand slot holding the SSA Value for the referenced array.}],
          "::mlir::OpOperand &", "getArrRefMutable", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Gets the type of the referenced array.
    inline ::llzk::array::ArrayType getArrRefType() { return getArrRef().getType(); }
  }];
}

def ArrayAccessOpInterface
    : OpInterface<"ArrayAccessOpInterface", [ArrayRefOpInterface]> {
  let description = [{
    Common interface for operations that read or write from an array.
  }];
  let cppNamespace = "::llzk::array";

  let methods = [
      // Requires implementors to have a '$indices' argument.
      InterfaceMethod<
          [{Gets the operand range containing the index for each dimension.}],
          "::mlir::Operation::operand_range", "getIndices", (ins)>,
      InterfaceMethod<
          [{Gets the mutable operand range containing the index for each dimension.}],
          "::mlir::MutableOperandRange", "getIndicesMutable", (ins)>,
      InterfaceMethod<
          [{Return `true` if the op is a read, `false` if it's a write.}],
          "bool", "isRead", (ins)>,
  ];

  let extraClassDeclaration = [{
    /// Compute the dimensions of the read/write value. Removing the number
    /// of indices provided from the front dimensions of the base array type
    /// gives the dimensions/shape of that value.
    inline ::mlir::ArrayRef<::mlir::Attribute> getValueOperandDims() {
      return getArrRefType().getDimensionSizes().drop_front(getIndices().size());
    }

    /// Returns the multi-dimensional indices of the array access as an Attribute
    /// array or a null pointer if a valid index cannot be computed for any dimension.
    ::mlir::ArrayAttr indexOperandsToAttributeArray();

    /// Required by companion interface DestructurableAccessorOpInterface / SROA pass
    bool canRewire(const ::mlir::DestructurableMemorySlot &slot,
            ::llvm::SmallPtrSetImpl<::mlir::Attribute> &usedIndices,
            ::mlir::SmallVectorImpl<::mlir::MemorySlot> &mustBeSafelyUsed,
            const ::mlir::DataLayout &dataLayout);

    /// Required by companion interface DestructurableAccessorOpInterface / SROA pass
    ::mlir::DeletionKind rewire(const ::mlir::DestructurableMemorySlot &slot,
            ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &subslots,
            ::mlir::OpBuilder &builder, const ::mlir::DataLayout &dataLayout);

    /// Generate `arith.constant` indices for one static array element position.
    static ::llvm::SmallVector<::mlir::Value>
    genIndexConstants(::mlir::OpBuilder &bldr, ::mlir::Location loc, ::mlir::ArrayAttr index);

    /// Create an `array.read` or `array.extract` for one concrete element or subarray.
    static ::mlir::Value
    genRead(::mlir::OpBuilder &bldr, ::mlir::Location loc, ::mlir::Value arrayRef,
            ::mlir::ValueRange indices);

    /// Create an `array.read` or `array.extract` for one concrete element or subarray.
    static ::mlir::Value
    genRead(::mlir::OpBuilder &bldr, ::mlir::Location loc, ::mlir::Value arrayRef,
            ::mlir::ArrayAttr index);

    /// Create an `array.write` or `array.insert` for one concrete element or subarray.
    static void genWrite(::mlir::OpBuilder &bldr, ::mlir::Location loc, ::mlir::Value arrayRef,
            ::mlir::ValueRange indices, ::mlir::Value value);

    /// Create an `array.write` or `array.insert` for one concrete element or subarray.
    static void genWrite(::mlir::OpBuilder &bldr, ::mlir::Location loc, ::mlir::Value arrayRef,
            ::mlir::ArrayAttr index, ::mlir::Value value);
  }];
}

#endif // LLZK_ARRAY_OP_INTERFACES
