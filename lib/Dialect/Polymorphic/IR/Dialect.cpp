//===-- Dialect.cpp - Polymorphic dialect implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Polymorphic/IR/Dialect.h"

#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Polymorphic/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Polymorphic/IR/Types.cpp.inc"

//===------------------------------------------------------------------===//
// PolymorphicDialect
//===------------------------------------------------------------------===//

mlir::Operation *llzk::polymorphic::PolymorphicDialect::materializeConstant(
    mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type, mlir::Location loc
) {
  if (llvm::isa<mlir::IndexType, mlir::IntegerType>(type)) {
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value)) {
      return builder.create<mlir::arith::ConstantOp>(loc, intAttr);
    }
  }
  return nullptr;
}

auto llzk::polymorphic::PolymorphicDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Polymorphic/IR/Ops.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Polymorphic/IR/Types.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<PolymorphicDialect>>();
}
