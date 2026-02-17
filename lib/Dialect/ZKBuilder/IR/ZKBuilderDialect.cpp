//===-- ZKBuilderDialect.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.cpp.inc"
#define GET_OP_CLASSES
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.cpp.inc"

namespace llzk::zkbuilder {

void ZKBuilderDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.cpp.inc"
      >();
}

} // namespace llzk::zkbuilder
