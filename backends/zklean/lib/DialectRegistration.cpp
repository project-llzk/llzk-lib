//===-- DialectRegistration.cpp - Register ZKLean dialects ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "zklean/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"
#include "zklean/DialectRegistration.h"

#include <mlir/IR/DialectRegistry.h>

namespace zklean {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<llzk::zkbuilder::ZKBuilderDialect, llzk::zkexpr::ZKExprDialect,
                  llzk::zkleanlean::ZKLeanLeanDialect>();
}
} // namespace zklean
