//===-- Dialect.cpp - R1CS dialect implementation ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Dialect.h"

#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"

#include <mlir/IR/DialectImplementation.h>

// TableGen'd implementation files
#include "r1cs/Dialect/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "r1cs/Dialect/IR/Attrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "r1cs/Dialect/IR/Types.cpp.inc"

void r1cs::R1CSDialect::initialize() {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "r1cs/Dialect/IR/Ops.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "r1cs/Dialect/IR/Types.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "r1cs/Dialect/IR/Attrs.cpp.inc"
  >();
  // clang-format on
}
