//===-- Dialect.cpp - Boolean dialect implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Bool/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes for below
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Bool/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// BoolDialect
//===------------------------------------------------------------------===//

mlir::Operation *llzk::boolean::BoolDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type, mlir::Location loc
) {
  // Materialize i1 constants (results of folded bool and cmp ops) as
  // arith.constant ops, which is already used alongside this dialect.
  if (llvm::isa<mlir::IntegerAttr>(value) && llvm::isa<mlir::IntegerType>(type) &&
      llvm::cast<mlir::IntegerType>(type).isInteger(1)) {
    return builder.create<mlir::arith::ConstantOp>(loc, llvm::cast<mlir::IntegerAttr>(value));
  }
  return nullptr;
}

auto llzk::boolean::BoolDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Bool/IR/Ops.cpp.inc"
  >();

  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/Bool/IR/Attrs.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<BoolDialect>>();
}
