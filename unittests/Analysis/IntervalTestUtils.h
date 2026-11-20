//===-- IntervalTestUtils.h - Interval analysis test utilities ---*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/DataFlowFramework.h>

#include <functional>
#include <gtest/gtest.h>
#include <utility>

namespace llzk::test {

/// Run ordered test visits during solver initialization so lattice updates can propagate.
/// Only this driver is initialized; the solver then drains dependencies queued by the visits.
class IntervalTestInitialization : public mlir::DataFlowAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntervalTestInitialization)

  /// Own the callback that establishes the test's initial visitation order.
  IntervalTestInitialization(mlir::DataFlowSolver &dataflowSolver, std::function<void()> callback)
      : DataFlowAnalysis(dataflowSolver), initializeCallback(std::move(callback)) {}

  /// Execute the callback while the solver is running, stopping on fatal test assertions.
  mlir::LogicalResult initialize(mlir::Operation *) override {
    initializeCallback();
    return mlir::success(!testing::Test::HasFatalFailure());
  }

  /// The driver creates no dependencies of its own.
  mlir::LogicalResult visit(mlir::ProgramPoint *) override { return mlir::success(); }

private:
  std::function<void()> initializeCallback;
};

} // namespace llzk::test
