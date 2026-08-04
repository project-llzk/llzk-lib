//===-- Transforms.cpp - C impl for transformation passes -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKTransformationPasses.capi.h.inc"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/CAPI/Pass.h>

using namespace llzk;

static inline void registerLLZKTransformationPasses() { registerTransformationPasses(); }

// Impl
#include "llzk/Transforms/LLZKTransformationPasses.capi.cpp.inc"
