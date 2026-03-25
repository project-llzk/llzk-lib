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
#include "llzk/Transforms/Parsers.h"

namespace pcl::conversion {

std::unique_ptr<mlir::Pass> createPCLLoweringPass();

#define GEN_PASS_REGISTRATION
#include "pcl-conv/Transforms/TransformationPasses.h.inc"

}; // namespace pcl::conversion
