//===-- Dialect.cpp - String dialect implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/String/IR/Dialect.h"

#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/String/IR/Types.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/String/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/String/IR/Types.cpp.inc"

//===------------------------------------------------------------------===//
// StringDialect
//===------------------------------------------------------------------===//

auto llzk::string::StringDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/String/IR/Ops.cpp.inc"
  >();

  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(clang-analyzer-core.StackAddressEscape)
  addTypes<
    #define GET_TYPEDEF_LIST
    #include "llzk/Dialect/String/IR/Types.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<StringDialect>>();
}
