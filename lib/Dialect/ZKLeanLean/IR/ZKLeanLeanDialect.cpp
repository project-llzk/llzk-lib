//===-- ZKLeanLeanDialect.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h"
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.h"

#include <mlir/IR/Builders.h>
#include <mlir/Support/LogicalResult.h>
#include <llvm/ADT/TypeSwitch.h>

#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.cpp.inc"
#define GET_OP_CLASSES
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.cpp.inc"

auto llzk::zkleanlean::StructDefOp::verifyRegions()
    -> mlir::LogicalResult {
  mlir::Region &body = getBodyRegion();
  if (!body.hasOneBlock())
    return emitOpError("expected body region with a single block");

  for (mlir::Operation &op : body.front()) {
    if (!mlir::isa<llzk::zkleanlean::MemberDefOp>(op)) {
      op.emitOpError(
          "only 'ZKLeanLean.member' ops are allowed in a 'ZKLeanLean.structure' body");
      return mlir::failure();
    }
  }

  return mlir::success();
}

auto llzk::zkleanlean::ZKLeanLeanDialect::initialize() -> void {
  addTypes<
#define GET_TYPEDEF_LIST
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "llzk/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.cpp.inc"
      >();
}
