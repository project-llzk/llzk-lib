//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Defines dialect and pass registration functions for ZKLean backend.
//
//===----------------------------------------------------------------------===//

#include "zklean/DialectRegistration.h"

#include "zklean/Conversions/Passes.h"
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "zklean/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>

namespace zklean {

void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      llzk::zkbuilder::ZKBuilderDialect,
      llzk::zkexpr::ZKExprDialect,
      llzk::zkleanlean::ZKLeanLeanDialect,
      mlir::func::FuncDialect
      // clang-format on
      >();
}

void registerPasses(mlir::DialectRegistry &) { zklean::registerConversionPasses(); }

} // namespace zklean
