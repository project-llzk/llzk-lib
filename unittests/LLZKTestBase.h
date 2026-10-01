//===-- LLZKTestBase.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/DialectRegistration.h"
#include "llzk/Dialect/Shared/Builders.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>

#include <gtest/gtest.h>

class LLZKTest : public ::testing::Test {
protected:
  mlir::MLIRContext ctx;
  mlir::Location loc;

  LLZKTest() : ctx(), loc(llzk::getUnknownLoc(&ctx)) {
    mlir::DialectRegistry registry;
    llzk::registerAllDialects(registry);
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
  }
};
