//===-- Dialect.cpp - Cast dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Dialect.h"

#include "llzk/Dialect/Cast/IR/Attrs.h"
#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Cast/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Cast/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// CastDialect
//===------------------------------------------------------------------===//

/// Materialize folded cast results as felt or arithmetic constants.
mlir::Operation *llzk::cast::CastDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type, mlir::Location loc
) {
  if (auto attr = llvm::dyn_cast<llzk::felt::FeltConstAttr>(value);
      attr && attr.getType() == type) {
    return llzk::felt::FeltConstantOp::create(builder, loc, attr);
  }
  // No need to check `value.getType() == type` because it's already done internally
  // by the `materialize()` function, returning `nullptr` on mismatch.
  return mlir::arith::ConstantOp::materialize(builder, value, type, loc);
}

auto llzk::cast::CastDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Cast/IR/Ops.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/Cast/IR/Attrs.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<CastDialect>>();
}
