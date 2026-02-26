//===-- DialectRegistration.cpp - Register ZKLean dialects ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "zklean/DialectRegistration.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>

#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "zklean/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"

namespace zklean {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      llzk::zkbuilder::ZKBuilderDialect, 
      llzk::zkexpr::ZKExprDialect,
      llzk::zkleanlean::ZKLeanLeanDialect,
      mlir::func::FuncDialect
      // clang-format on
      >();
}
} // namespace zklean
