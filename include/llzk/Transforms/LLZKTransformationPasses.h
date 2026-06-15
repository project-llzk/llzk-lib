//===-- LLZKTransformationPasses.h ------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Config/Config.h"
#include "llzk/Pass/PassBase.h"

#include <llvm/ADT/StringRef.h>

namespace llzk {

void addRemoveUnnecessaryOpsAndDefsPipeline(mlir::OpPassManager &pm);

void registerTransformationPassPipelines();

void registerInliningExtensions(mlir::DialectRegistry &registry);

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"

} // namespace llzk
