//===-- Passes.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Config/Config.h"
#include "llzk/Pass/PassBase.h"

namespace llzk {

std::unique_ptr<mlir::Pass> createFeltToLLVMConversionPass();

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "llzk/Conversions/Passes.h.inc"

} // namespace llzk
