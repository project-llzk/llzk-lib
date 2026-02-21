//===-- ZKBuilderDialect.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderOps.h"
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderTypes.h"
#include "zklean/Dialect/ZKExpr/IR/ZKExprTypes.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/TypeSwitch.h>

// Include TableGen'd declarations
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderTypes.cpp.inc"
#define GET_OP_CLASSES
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderOps.cpp.inc"

namespace llzk::zkbuilder {

void ZKBuilderDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderOps.cpp.inc"
      >();
}

} // namespace llzk::zkbuilder
