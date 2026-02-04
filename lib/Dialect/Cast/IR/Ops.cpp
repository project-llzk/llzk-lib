//===-- Ops.cpp - Cast operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Cast/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Cast/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::component;
using namespace llzk::function;

namespace llzk::cast {

//===------------------------------------------------------------------===//
// FeltToIndexOp
//===------------------------------------------------------------------===//

LogicalResult FeltToIndexOp::verify() {
  if (auto parentOr = getParentOfType<FuncDefOp>(*this);
      succeeded(parentOr) && !parentOr->hasAllowWitnessAttr()) {
    // Traverse the def-use chain to see if this operand, which is a felt, ever
    // derives from a Signal struct.
    SmallVector<Value, 2> frontier {getValue()};
    DenseSet<Value> visited;

    while (!frontier.empty()) {
      Value v = frontier.pop_back_val();
      if (visited.contains(v)) {
        continue;
      }
      visited.insert(v);

      if (Operation *op = v.getDefiningOp()) {
        frontier.insert(frontier.end(), op->operand_begin(), op->operand_end());
      }
    }
  }

  return success();
}

} // namespace llzk::cast
