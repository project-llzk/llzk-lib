//===-- ZKLeanPasses.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Pass/Pass.h>

#include <memory>

namespace zklean {

/// Emits a textual dump of zk-related dialect operations to help debugging
/// pipelines that mix zkLean, zkBuilder, and zkExpr IR.
std::unique_ptr<mlir::Pass> createPrettyPrintZKLeanPass();

#define GEN_PASS_REGISTRATION
#include "zklean/Transforms/ZKLeanPasses.h.inc"

} // namespace zklean
