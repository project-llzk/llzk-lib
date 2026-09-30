//===-- SMTInfoDialect.cpp - SMT script metadata dialect --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.h"

#include "llzk/Dialect/SMTInfo/IR/SMTInfoOps.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

// TableGen'd implementation files
#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.cpp.inc"

using namespace mlir;
using namespace llzk::smt_info;

void SMTInfoDialect::initialize() {
  registerAttributes();
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/SMTInfo/IR/SMTInfoOps.cpp.inc"
  >();
  // clang-format on
}
