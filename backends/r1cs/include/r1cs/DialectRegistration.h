//===-- DialectRegistration.h -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Declares dialect and pass registration functions for R1CS backend.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace r1cs {

/// Register the R1CS dialect in \p registry.
void registerDialects(mlir::DialectRegistry &registry);

/// Register R1CS transformation passes and pipelines in the global pass registry.
void registerPasses(mlir::DialectRegistry &registry);

} // namespace r1cs
