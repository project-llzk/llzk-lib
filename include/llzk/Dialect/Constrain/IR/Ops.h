//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/OpInterfaces.h"
#include "llzk/Dialect/Function/IR/OpTraits.h"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Constrain/IR/Ops.h.inc"

namespace llzk::constrain {

/// Check the complete constrain.in operand contract without emitting diagnostics, including
/// trailing dimensions, element unification, and the exclusion of struct/string elements.
bool areValidContainmentOperandTypes(mlir::Type lhsType, mlir::Type rhsType);

} // namespace llzk::constrain
