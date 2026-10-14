//===-- DialectRegistration.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Declares dialect and pass registration functions for PCL backend.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace pcl {

/// Register the PCL and MLIR func dialects in \p registry.
void registerDialects(mlir::DialectRegistry &registry);

/// Register PCL conversion and transformation passes in the global pass registry.
void registerPasses(mlir::DialectRegistry &registry);

} // namespace pcl
