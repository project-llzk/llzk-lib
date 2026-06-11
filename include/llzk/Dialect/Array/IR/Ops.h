//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Interfaces/InferTypeOpInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace llzk::array {

/// Memory resource for allocations that may be erased when no stored value is ever read.
struct DiscardableAllocationResource
    : public mlir::SideEffects::Resource::Base<DiscardableAllocationResource> {
  mlir::StringRef getName() final { return "DiscardableAllocation"; }
};

} // namespace llzk::array

// Include TableGen'd declarations
#include "llzk/Dialect/Array/IR/OpInterfaces.h.inc"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Array/IR/Ops.h.inc"
