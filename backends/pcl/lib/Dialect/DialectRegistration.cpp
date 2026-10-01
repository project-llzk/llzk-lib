//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Defines dialect and pass registration functions for PCL backend.
//
//===----------------------------------------------------------------------===//

#include "pcl/DialectRegistration.h"

#include "pcl/Dialect/IR/Dialect.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>

namespace pcl {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, PCLDialect>();
}

} // namespace pcl
