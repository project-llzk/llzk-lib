//===-- Dialect.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

// Include TableGen'd declarations
#include "llzk/Dialect/Verif/IR/Dialect.h.inc"

namespace llzk::verif {
/// Attaches the interfaces defined by the `verif` dialect to upstream IR elements.
///
/// Attempting to use those interfaces without calling this function first will result in an error.
void attachInterfaces(mlir::MLIRContext &context);

/// Registers dialect extensions for the verif dialect.
void registerExtensions(mlir::DialectRegistry &registry);
} // namespace llzk::verif
