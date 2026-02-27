//===-- LLZKEnforceNoOverwritePass.cpp --------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-enforce-no-overwrite` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <memory>

namespace llzk {
#define GEN_PASS_DECL_ENFORCENOMEMBEROVERWRITEPASS
#define GEN_PASS_DEF_ENFORCENOMEMBEROVERWRITEPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

#define DEBUG_TYPE "llzk-enforce-no-overwrites-pass"

class EnforceNoMemberOverwritePass
    : public llzk::impl::EnforceNoMemberOverwritePassBase<EnforceNoMemberOverwritePass> {
  void runOnOperation() override {}
};

namespace llzk {
using std::make_unique;

std::unique_ptr<mlir::Pass> createNoOverwritesPass() {
  return make_unique<EnforceNoMemberOverwritePass>();
}
} // namespace llzk
