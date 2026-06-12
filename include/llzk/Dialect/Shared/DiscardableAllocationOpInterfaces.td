//===-- DiscardableAllocationOpInterfaces.td --------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_DISCARDABLE_ALLOCATION_OP_INTERFACES
#define LLZK_DISCARDABLE_ALLOCATION_OP_INTERFACES

include "mlir/IR/Interfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DiscardableAllocationResource
    : Resource<"::llzk::DiscardableAllocationResource">;

def DiscardableAllocationAccessorOpInterface
    : OpInterface<"DiscardableAllocationAccessorOpInterface"> {
  let description = [{
    Provides the necessary information to the `RemoveUnusedDiscardableAllocations` pass about
    operations that load or store values from the result of an allocator op marked with the
    `MemAlloc<DiscardableAllocationResource>` memory effect so unused allocators and stores can be
    removed.
  }];
  let cppNamespace = "::llzk";

  let methods =
      [InterfaceMethod<
           [{Returns true if this op reads from the allocation represented by `ptr`.}],
           "bool", "loadsFromDiscardableAllocation",
           (ins "::mlir::Value":$ptr)>,
       InterfaceMethod<
           [{Returns true if this op writes to the allocation represented by `ptr`.}],
           "bool", "storesToDiscardableAllocation", (ins "::mlir::Value":$ptr)>,
       InterfaceMethod<
           [{Returns true if this op can be erased as a dead store to `ptr`.}],
           "bool", "canEraseAsDeadStoreTo",
           (ins "::mlir::Value":$ptr,
               "const ::mlir::DataLayout &":$dataLayout)>,
  ];
}

#endif // LLZK_DISCARDABLE_ALLOCATION_OP_INTERFACES
