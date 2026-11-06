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
/// Declares dialect and pass registration functions for SMT backend.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace llzk::smt {

/// Register the MLIR smt and LLZK smt_info metadata dialects in \p registry.
void registerDialects(mlir::DialectRegistry &registry);

/// Register SMT conversion passes in the global pass registry.
/// Add the MLIR func inlining extension to \p registry for SMT lowering.
void registerPasses(mlir::DialectRegistry &registry);

} // namespace llzk::smt
