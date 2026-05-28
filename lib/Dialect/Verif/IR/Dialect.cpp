//===-- Dialect.cpp - Verif dialect implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/IR/Dialect.h"

#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir/IR/DialectImplementation.h>

#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Verif/IR/Dialect.cpp.inc"

//===------------------------------------------------------------------===//
// VerifDialect
//===------------------------------------------------------------------===//

auto llzk::verif::VerifDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Verif/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<VerifDialect>>();
}
