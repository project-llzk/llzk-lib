//===-- Dialect.cpp - Array dialect implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Array/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Array/IR/Types.cpp.inc"

//===------------------------------------------------------------------===//
// ArrayDialect
//===------------------------------------------------------------------===//

auto llzk::array::ArrayDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Array/IR/Ops.cpp.inc"
  >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/Array/IR/Types.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<ArrayDialect>>();
}
