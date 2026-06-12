//===-- DiscardableAllocationOpInterfaces.h ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace llzk {

/// Memory resource for allocations that may be erased when no stored value is ever read.
struct DiscardableAllocationResource
    : public mlir::SideEffects::Resource::Base<DiscardableAllocationResource> {
  mlir::StringRef getName() final;
};

} // namespace llzk

// Include TableGen'd declarations
#include "llzk/Transforms/DiscardableAllocationOpInterfaces.h.inc"
