//===-- Analysis.cpp - C impl for analysis passes ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Analysis.h"

#include "llzk/Analysis/AnalysisPasses.h"

#include <mlir/CAPI/Pass.h>

using namespace llzk;

/// Adapt the C API group name to the C++ analysis pass registration function.
static inline void registerLLZKAnalysisPasses() { registerAnalysisPasses(); }

// Impl
#include "llzk/Analysis/AnalysisPasses.capi.cpp.inc"
