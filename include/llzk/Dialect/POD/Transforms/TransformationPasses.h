//===-- TransformationPasses.h ----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Pass/PassBase.h"

namespace llzk::pod {

std::unique_ptr<mlir::Pass> createPodToScalarPass();

#define GEN_PASS_REGISTRATION
#include "llzk/Dialect/POD/Transforms/TransformationPasses.h.inc"

}; // namespace llzk::pod
