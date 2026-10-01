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

#include "pcl/Conversion/ConversionPasses.h"
#include "pcl/Dialect/IR/Dialect.h"
#include "pcl/Transforms/TransformationPasses.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/DialectRegistry.h>

namespace pcl {

void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect, PCLDialect>();
}

void registerPasses(mlir::DialectRegistry &) {
  pcl::registerPCLConversionPasses();
  pcl::registerTransformationPasses();
}

} // namespace pcl
