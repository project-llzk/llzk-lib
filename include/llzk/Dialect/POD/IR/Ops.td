//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POD_OPS
#define LLZK_POD_OPS

include "llzk/Dialect/POD/IR/Dialect.td"
include "llzk/Dialect/POD/IR/OpInterfaces.td"
include "llzk/Dialect/POD/IR/Types.td"
include "llzk/Dialect/Shared/OpTraits.td"
include "llzk/Dialect/Shared/DiscardableAllocationOpInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class PODDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<PODDialect, mnemonic, traits>;

class PODAccessOpBase<string mnemonic, list<Trait> traits = []>
    : PODDialectOp<mnemonic,
                   traits#[DeclareOpInterfaceMethods<PodAccessOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the type of the referenced pod.
    inline ::llzk::pod::PodType getPodRefType() {
      return ::llvm::cast<PodAccessOpInterface>(getOperation()).getPodRefType();
    }
  }];
}

// isRead: read(1) vs write(0) ops
class ScalarPODAccessOp<string mnemonic, bit isRead, list<Trait> traits = []>
    : PODAccessOpBase<
          mnemonic,
          traits#[DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
                  DeclareOpInterfaceMethods<PromotableMemOpInterface>,
                  DeclareOpInterfaceMethods<
                      DiscardableAllocationAccessorOpInterface>]> {
  let extraClassDefinition = [{
    /// Required by DestructurableAllocationOpInterface / SROA pass
    bool $cppClass::canRewire(const ::mlir::DestructurableMemorySlot &slot,
                ::llvm::SmallPtrSetImpl<::mlir::Attribute> &usedIndices,
                ::mlir::SmallVectorImpl<::mlir::MemorySlot> &mustBeSafelyUsed,
                const ::mlir::DataLayout &dataLayout) {
      return ::llvm::cast<PodAccessOpInterface>(getOperation())
                .canRewire(slot, usedIndices, mustBeSafelyUsed, dataLayout);
    }

    /// Required by DestructurableAllocationOpInterface / SROA pass
    ::mlir::DeletionKind $cppClass::rewire(const ::mlir::DestructurableMemorySlot &slot,
                ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &subslots,
                ::mlir::OpBuilder &builder, const ::mlir::DataLayout &dataLayout) {
      return ::llvm::cast<PodAccessOpInterface>(getOperation())
                .rewire(slot, subslots, builder, dataLayout);
    }

    /// Required by PromotableMemOpInterface / mem2reg pass
    bool $cppClass::loadsFrom(const ::mlir::MemorySlot &slot) {
      return }]#!if(isRead, "getPodRef() == slot.ptr", "false")#[{;
    }

    /// Required by PromotableMemOpInterface / mem2reg pass
    bool $cppClass::storesTo(const ::mlir::MemorySlot &slot) {
      return }]#!if(isRead, "false", "getPodRef() == slot.ptr")#[{;
    }

    /// Required by PromotableAllocationOpInterface / mem2reg pass
    ::mlir::Value $cppClass::getStored(const ::mlir::MemorySlot &, ::mlir::OpBuilder &,
                                       ::mlir::Value, const ::mlir::DataLayout &) {
      }]#!if(
      isRead,
      "llvm_unreachable(\"getStored() should not be called on $cppClass\")",
      "return getValue()")#[{;
    }

    /// Required by PromotableMemOpInterface / mem2reg pass
    bool $cppClass::canUsesBeRemoved(
        const ::mlir::MemorySlot &slot,
        const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &blockingUses,
        ::llvm::SmallVectorImpl<::mlir::OpOperand *> & /*newBlockingUses*/,
        const ::mlir::DataLayout & /*datalayout*/) {
      if (blockingUses.size() != 1) {
        return false;
      }
      ::mlir::Value blockingUse = (*blockingUses.begin())->get();
      return blockingUse == slot.ptr && getPodRef() == slot.ptr &&
      }]#!if(isRead, "getResult().getType() == slot.elemType",
             "getValue() != slot.ptr && getValue().getType() == slot.elemType")#[{;
    }

    /// Required by PromotableMemOpInterface / mem2reg pass
    ::mlir::DeletionKind $cppClass::removeBlockingUses(
        const ::mlir::MemorySlot &, const ::llvm::SmallPtrSetImpl<::mlir::OpOperand *> &,
        ::mlir::OpBuilder &, ::mlir::Value reachingDefinition, const ::mlir::DataLayout &) {
      }]#!if(isRead, "getResult().replaceAllUsesWith(reachingDefinition)",
             "")#[{;
      return ::mlir::DeletionKind::Delete;
    }

    /// Return `true` if the op is a read, `false` if it's a write.
    bool $cppClass::isRead() {
      return }]#!if(isRead, "true", "false")#[{;
    }

    /// Required by DiscardableAllocationAccessorOpInterface / unused allocation cleanup.
    bool $cppClass::loadsFromDiscardableAllocation(::mlir::Value ptr) {
      return }]#!if(isRead, "getPodRef() == ptr", "false")#[{;
    }

    /// Required by DiscardableAllocationAccessorOpInterface / unused allocation cleanup.
    bool $cppClass::storesToDiscardableAllocation(::mlir::Value ptr) {
      return }]#!if(isRead, "false",
                    "getPodRef() == ptr && getValue() != ptr")#[{;
    }

    /// Required by DiscardableAllocationAccessorOpInterface / unused allocation cleanup.
    bool $cppClass::canEraseAsDeadStoreTo(::mlir::Value ptr, const ::mlir::DataLayout &) {
      return }]#!if(isRead, "false",
                    "getPodRef() == ptr && getValue() != ptr")#[{;
    }
  }];
}

def LLZK_NewPodOp
    : PODDialectOp<
          "new", [MemoryEffects<[MemAlloc<DiscardableAllocationResource>]>,
                  AttrSizedOperandSegments, VerifySizesForMultiAffineOps<1>,
                  DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
                  DeclareOpInterfaceMethods<
                      DestructurableAllocationOpInterface>,
                  DeclareOpInterfaceMethods<
                      OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "create a new plain-old-data struct";
  let description = [{
    Creates a new, uninitialized, pod instance. Optionally, the user can pass a list of record names and values
    that initialize the records of the pod. Partial initialization is allowed. All records without
    an explicit initialization are initialized with nondeterministic values.

    If the types of the pod records have affine map parameters the user can pass values for them similar
    to how `array.new` does it.

    This operation returns one value of type `PODType` and, if present, the records passed for initialization
    must form a subset of the records in the type.

    Examples:
    ```llzk
    // Empty pod instance
    %0 = pod.new : !pod.type<[]>

    // Uninitialized/nondeterministic pod instance
    %0 = pod.new : !pod.type<[@n: !felt.type]>

    // Initialized pod instance
    %0 = felt.const 1
    %1 = pod.new { @n = %0 } : !pod.type<[@n: !felt.type]>

    // Another one, but with 2 fields
    %0 = felt.const 1
    %1 = felt.inv %0
    %2 = pod.new { @n = %0, @inv = %1 } :	!pod.type<[@n: !felt.type, @inv: !felt.type]>

    // Partial init
    %0 = felt.const 1
    %1 = pod.new { @n = %0 } : !pod.type<[@n: !felt.type, @inv: !felt.type]>

    // Affine map args on uninitialized pod instance
    %0 = arith.constant 1 : index
    %c = arith.constant 2 : index
    %1 = pod.new(%0)[%c] : !pod.type<[@a: !array.type<#map, !felt.type>]>

    // Affine map with initialized records
    %0 = arith.constant 1 : index
    %c = arith.constant 2 : index
    %1 = felt.const 5
    %2 = pod.new { @f = %1 }(%0)[%c] : !pod.type<[@f: !felt.type, @a: !array.type<#map, !felt.type>]>
    ```
  }];

  let arguments = (ins
      // Initialization values
      Variadic<AnyLLZKType>:$initialValues,
      DefaultValuedAttr<StrArrayAttr, "{}">:$initializedRecords,
      // Affine map arguments
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs LLZK_PODType:$result);
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<
           (ins CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues),
           [{
        auto resultType = ::llzk::pod::PodType::fromInitialValues($_builder.getContext(), initialValues);
        build($_builder, $_state, resultType, initialValues);
      }]>,
       OpBuilder<(ins "::llzk::pod::PodType":$resultType,
           CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues)>,
       OpBuilder<(ins "::llzk::pod::PodType":$resultType,
           "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
           "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
           CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues)>,
       OpBuilder<
           (ins "::llzk::pod::PodType":$resultType,
               "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
               "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
               CArg<"::llzk::pod::InitializedRecords", "{}">:$initialValues),
           [{
      build($_builder, $_state, resultType, mapOperands,
         $_builder.getDenseI32ArrayAttr(numDimsPerMap), initialValues);
    }]>];
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::SmallVector<::llzk::pod::RecordValue> getInitializedRecordValues();
  }];
}

def LLZK_ReadPodOp : ScalarPODAccessOp<"read", 1> {
  let summary = "reads the contents of a plain-old-data struct record";
  let description = [{
    Reads the current value of a named record from a pod instance.
    Returns one value of the type of the record.

    The name of the record must be a valid record name for the pod type and the result
    type must match the type of the record.

    Example:
    ```llzk
    %1 = pod.read %0[@sym] : !pod.type<[@sym: !type, ...]>, !type
    ```
  }];

  let arguments = (ins Arg<LLZK_PODType, "the pod to read from">:$pod_ref,
      StrAttr:$record_name);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{
    $pod_ref `[` custom<RecordName>($record_name) `]` `:` type($pod_ref) `,` type($result) attr-dict
  }];
  let hasVerifier = 1;
}

def LLZK_WritePodOp : ScalarPODAccessOp<"write", 0> {
  let summary = "writes content into a plain-old-data struct record";
  let description = [{
    Writes a value into a named record of a pod instance. This operation has no result values.

    The name of the record must be a valid record name for the pod type and the source
    value type must match the type of the record.

    Writing a record that was written or initialized earlier overwrites the previous value.

    Example:
    ```llzk
    pod.write %0[@sym] = %1 : !pod.type<[@sym: !type, ...]>, !type
    ```
  }];

  let arguments = (ins Arg<LLZK_PODType, "the pod to write into">:$pod_ref,
      StrAttr:$record_name, AnyLLZKType:$value);

  let assemblyFormat = [{
    $pod_ref `[` custom<RecordName>($record_name) `]` `=` $value `:` type($pod_ref) `,` type($value) attr-dict
  }];
  let hasVerifier = 1;
}

#endif // LLZK_POD_OPS
