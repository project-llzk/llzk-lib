//===-- Dialect.cpp - Constrain dialect implementation ----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Constrain/IR/Dialect.h"

#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

// TableGen'd implementation files
#include "llzk/Dialect/Constrain/IR/Dialect.cpp.inc"

//===------------------------------------------------------------------===//
// ConstrainDialect
//===------------------------------------------------------------------===//

auto llzk::constrain::ConstrainDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Constrain/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<ConstrainDialect>>();
}
