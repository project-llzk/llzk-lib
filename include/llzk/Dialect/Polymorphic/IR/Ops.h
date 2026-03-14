//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Shared/OpHelpers.h"

#include <mlir/Dialect/Affine/IR/AffineValueMap.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

// Include TableGen'd declarations
#include "llzk/Dialect/Polymorphic/IR/OpInterfaces.h.inc"

namespace llzk::polymorphic {

template <typename OpT>
concept ConstParamSymbol = std::is_same_v<OpT, ConstParamSymbolOpInterface> ||
                           std::is_base_of_v<ConstParamSymbolOpInterface::Trait<OpT>, OpT>;
}

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Polymorphic/IR/Ops.h.inc"

namespace llzk::polymorphic {

/// Return true iff the given Operation is nested somewhere within a TemplateOp.
bool isInTemplate(mlir::Operation *op);

/// If the given Operation is nested somewhere within a TemplateOp, return a success result
/// containing that TemplateOp. Otherwise emit an error and return a failure result.
mlir::FailureOr<TemplateOp> verifyInTemplate(mlir::Operation *op);

} // namespace llzk::polymorphic
