//===-- Passes.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"

namespace llzk {

std::unique_ptr<mlir::Pass> createConvertFeltConstToZKExprPass();

/// Registers all conversion passes defined in this directory.
void registerConversionPasses();

} // namespace llzk

#define GEN_PASS_DECL_CONVERTFELTCONSTTOZKEXPRPASS
#include "llzk/Conversions/LLZKConversionPasses.h.inc"
