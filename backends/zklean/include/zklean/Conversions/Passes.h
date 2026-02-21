//===-- Passes.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"

namespace llzk {

std::unique_ptr<mlir::Pass> createConvertLLZKToZKLeanPass();
std::unique_ptr<mlir::Pass> createConvertZKLeanToLLZKPass();

#define GEN_PASS_REGISTRATION
#include "zklean/Conversions/LLZKConversionPasses.h.inc"

} // namespace llzk
