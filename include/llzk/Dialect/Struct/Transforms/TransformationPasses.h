//===-- TransformationPasses.h ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Pass/PassBase.h"

namespace llzk::component {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/Struct/Transforms/TransformationPasses.h.inc"

} // namespace llzk::component
