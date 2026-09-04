//===-- InitializerUtils.h - Global initializer normalization ---*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/ErrorHelper.h"

#include <mlir/IR/Attributes.h>

namespace llzk::global {

/// A global initializer and its normalized type.
struct NormalizedGlobalInitializer {
  mlir::Type type;
  mlir::Attribute value;
};

/// Normalize unambiguous initializer representations and their declared type.
///
/// Explicitly conflicting felt fields are preserved so GlobalDefOp verification
/// can reject them without this helper silently choosing one field.
mlir::FailureOr<NormalizedGlobalInitializer>
normalizeGlobalInitializer(mlir::Type type, mlir::Attribute value, EmitErrorFn emitError);

} // namespace llzk::global
