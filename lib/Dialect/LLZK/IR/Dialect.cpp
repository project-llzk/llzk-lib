//===-- Dialect.cpp - Dialect method implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Config/Config.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/Bytecode/BytecodeImplementation.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Support/LLVM.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/LLZK/IR/Dialect.cpp.inc"

// Need a complete declaration of storage classes for below
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"

//===------------------------------------------------------------------===//
// LLZKDialect
//===------------------------------------------------------------------===//

auto llzk::LLZKDialect::initialize() -> void {
  // clang-format off
  addAttributes<
    #define GET_ATTRDEF_LIST
    #include "llzk/Dialect/LLZK/IR/Attrs.cpp.inc"
  >();

  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<LLZKDialect>>();
}
