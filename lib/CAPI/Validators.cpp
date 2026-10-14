//===-- Validators.cpp - C impl for validation passes -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Validators/LLZKValidationPasses.capi.h.inc"
#include "llzk/Validators/LLZKValidationPasses.h"

#include <mlir/CAPI/Pass.h>

using namespace llzk;

static inline void registerLLZKValidationPasses() { registerValidationPasses(); }

// Impl
#include "llzk/Validators/LLZKValidationPasses.capi.cpp.inc"
