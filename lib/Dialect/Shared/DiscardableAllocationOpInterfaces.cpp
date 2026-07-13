//===-- DiscardableAllocationOpInterfaces.cpp -------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Shared/DiscardableAllocationOpInterfaces.h"

// Include the generated interface definitions.
#include "llzk/Dialect/Shared/DiscardableAllocationOpInterfaces.cpp.inc"

namespace llzk {

// DiscardableAllocationResource::DiscardableAllocationResource()
//     : mlir::SideEffects::Resource::Base<DiscardableAllocationResource>() {}
//
// // Note: definition is placed here rather than the header to avoid the error:
// //       "vtable will be emitted in every translation unit"
// mlir::StringRef DiscardableAllocationResource::getName() { return "DiscardableAllocation"; }

} // namespace llzk
