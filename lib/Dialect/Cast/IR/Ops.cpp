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

LogicalResult FeltToIndexOp::verifySymbolUses(SymbolTableCollection &tables) {
  if (auto parentOr = getParentOfType<FuncDefOp>(*this);
      succeeded(parentOr) && !parentOr->hasAllowNonNativeFieldOpsAttr()) {
    // Traverse the def-use chain to see if this operand, which is a felt,
    // derives from a member with the `signal` attribute.
    SmallVector<Value, 2> frontier {getValue()};
    DenseSet<Value> visited;

    while (!frontier.empty()) {
      Value v = frontier.pop_back_val();
      if (visited.contains(v)) {
        continue;
      }
      visited.insert(v);

      if (Operation *op = v.getDefiningOp()) {
        if (MemberReadOp read = llvm::dyn_cast<MemberReadOp>(op)) {
          auto readFrom = read.getMemberDefOp(tables);
          if (succeeded(readFrom) && readFrom->get().getSignal()) {
            return emitOpError()
                .append(
                    "input is derived from a '", MemberDefOp::getOperationName(),
                    "' with 'signal' attribute, which is only valid within a '",
                    FuncDefOp::getOperationName(), "' with '", AllowNonNativeFieldOpsAttr::name,
                    "' attribute"
                )
                .attachNote(read.getLoc())
                .append("signal value is read here");
          }
        }
        frontier.insert(frontier.end(), op->operand_begin(), op->operand_end());
      }
    }
  }
  return success();
}

} // namespace llzk::cast
