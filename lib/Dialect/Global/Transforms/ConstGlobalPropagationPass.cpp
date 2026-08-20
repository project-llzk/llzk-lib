//===-- ConstGlobalPropagationPass.cpp --------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-const-global-propagation` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h"

// Include the generated base pass class definitions.
namespace llzk::global {
#define GEN_PASS_DEF_CONSTGLOBALPROPAGATIONPASS
#include "llzk/Dialect/Global/Transforms/TransformationPasses.h.inc"
} // namespace llzk::global

using namespace mlir;
using namespace llzk;
using namespace llzk::global;

namespace {

class PassImpl : public llzk::global::impl::ConstGlobalPropagationPassBase<PassImpl> {
  using Base = ConstGlobalPropagationPassBase<PassImpl>;

public:
  using Base::Base;

  void runOnOperation() override {
    // TODO: find uses of const global symbols and replace them with their values.
  }
};

} // namespace
