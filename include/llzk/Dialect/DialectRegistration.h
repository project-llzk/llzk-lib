//===-- DialectRegistration.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Declares dialect and pass registration functions for core LLZK dialects.
///
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace llzk {

/// Register core LLZK dialects, the MLIR arith, scf, and smt dialects, and verification
/// extensions in \p registry.
void registerDialects(mlir::DialectRegistry &registry);

/// Register LLZK analysis, validation, and transformation passes and pipelines, SMT
/// conversion passes, and MLIR transformation passes in the global pass registry.
/// Add LLZK and MLIR func inlining extensions to \p registry.
void registerPasses(mlir::DialectRegistry &registry);

} // namespace llzk
