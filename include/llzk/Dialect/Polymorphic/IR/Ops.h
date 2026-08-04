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

#include <llvm/ADT/StringRef.h>

// Include TableGen'd declarations
#include "llzk/Dialect/Polymorphic/IR/OpInterfaces.h.inc"

namespace llzk::polymorphic {

/// Metadata carried across transformation passes preserving the literal chunks of a
/// partially-instantiated template's logical name.
///
/// When present, the attribute has P+1 StringAttr chunks around the P current template
/// parameters. The gap between chunks `i` and `i+1` represents current parameter `i`. The chunks
/// are literal data; later transforms never recover semantic positions from the physical symbol
/// name.
inline constexpr llvm::StringLiteral TEMPLATE_NAME_PATTERN_ATTR = "poly.name_pattern";

template <typename OpT>
concept TemplateSymbolBindingOp =
    std::is_same_v<OpT, TemplateSymbolBindingOpInterface> ||
    std::is_base_of_v<TemplateSymbolBindingOpInterface::Trait<OpT>, OpT>;
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
