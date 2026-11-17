//===-- FlatteningScalarization.h -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Private heterogeneous-array scalarization entry point.
///
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

namespace llzk::polymorphic::detail::flattening {

class ConversionTracker;

/// Scalarize safe heterogeneous arrays, including all writes and reads of their split members.
/// Invoke before general type propagation so per-index types are preserved.
mlir::LogicalResult scalarizeHeterogeneousArrays(mlir::ModuleOp modOp, ConversionTracker &tracker);

} // namespace llzk::polymorphic::detail::flattening
