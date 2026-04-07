//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Function/IR/Attrs.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

#include <cstdint>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/Function/IR/Ops.h.inc"

namespace llzk::function {

/// @brief Kinds of functions in LLZK.
enum class FunctionKind : std::uint8_t {
  /// @brief Function within a struct named `FUNC_NAME_COMPUTE`.
  StructCompute,
  /// @brief Function within a struct named `FUNC_NAME_CONSTRAIN`.
  StructConstrain,
  /// @brief Function within a struct named `FUNC_NAME_PRODUCT`.
  StructProduct,
  /// @brief Function that is not within a struct.
  Free
};

/// @brief Given a function name, return the corresponding `FunctionKind`.
///
/// One caveat to note is that this cannot check if the function is actually within a struct
/// (it only checks if the name is one of the special names for struct functions defined in
/// `include/llzk/Util/Constants.h`) so, regardless of the `FunctionKind` returned, the
/// function may still be a free function and additional checks may be necessary.
FunctionKind fnNameToKind(mlir::StringRef name);

} // namespace llzk::function
