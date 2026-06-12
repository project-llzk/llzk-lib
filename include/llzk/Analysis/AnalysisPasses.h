//===-- AnalysisPasses.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Analysis/AnalysisPassEnums.h"
#include "llzk/Pass/PassBase.h"

namespace llzk {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "llzk/Analysis/AnalysisPasses.h.inc"

} // namespace llzk
