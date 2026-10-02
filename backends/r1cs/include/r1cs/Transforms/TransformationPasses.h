//===-- TransformationPasses.h ----------------------------------*- C++ -*-===//
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

namespace r1cs {

/// Marks normalized constraints whose R1CS auxiliary witness assignments exist.
constexpr char PREPARED_ATTR_NAME[] = "r1cs.prepared";

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "r1cs/Transforms/TransformationPasses.h.inc"

} // namespace r1cs
