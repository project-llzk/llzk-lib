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
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/Dialect/Arith/IR/Arith.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Cast/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/Cast/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// CastDialect
//===------------------------------------------------------------------===//

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
