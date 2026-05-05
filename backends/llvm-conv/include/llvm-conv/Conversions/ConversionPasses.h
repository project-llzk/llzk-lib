//===-- ConversionPasses.h --------------------------------------*- C++ -*-===//
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
#include "llzk/Transforms/Parsers.h"

namespace llzk::llvm_conv {

std::unique_ptr<mlir::Pass> createLLVMLoweringPass();

#define GEN_PASS_REGISTRATION
#include "llvm-conv/Conversions/ConversionPasses.h.inc"

}; // namespace llzk::llvm_conv
