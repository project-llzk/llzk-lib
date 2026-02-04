//===- LightweightSignalEquivalenceAnalysis.cpp ---------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/lib/Analysis/DataFlow/SparseAnalysis.cpp.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a basic signal equivalence analysis
///
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/LightweightSignalEquivalenceAnalysis.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "llzk-signal-equivalence"

using namespace mlir;
using namespace llzk::component;

namespace llzk {

LightweightSignalEquivalenceAnalysis::LightweightSignalEquivalenceAnalysis(Operation *) {}

Value replaceReadWithWrite(Value v) {
  if (!v.getDefiningOp()) {
    return v;
  }
  if (auto read = dyn_cast<MemberReadOp>(v.getDefiningOp())) {
    // Traverse backwards through the block until we find a write
    for (Operation *cur = read; cur != nullptr; cur = cur->getPrevNode()) {
      if (auto write = dyn_cast<MemberWriteOp>(cur)) {
        // Return the written value
        return write.getVal();
      }
    }
  }
  return v;
}

bool LightweightSignalEquivalenceAnalysis::areSignalsEquivalent(Value v1, Value v2) {
  v1 = replaceReadWithWrite(v1);
  v2 = replaceReadWithWrite(v2);
  LLVM_DEBUG(llvm::outs() << "Asking for equivalence between " << v1 << " and " << v2 << "\n");
  if (equivalentSignals.isEquivalent(v1, v2)) {
    return true;
  }

  Operation *o1 = v1.getDefiningOp();
  Operation *o2 = v2.getDefiningOp();

  if (o1 == nullptr || o2 == nullptr) {
    return false;
  }

  if (o1->getName() != o2->getName()) {
    return false;
  }

  if (o1->getNumOperands() != o2->getNumOperands()) {
    return false;
  }

  for (size_t i = 0; i < o1->getNumOperands(); i++) {
    if (!areSignalsEquivalent(o1->getOperand(i), o2->getOperand(i))) {
      return false;
    }
  }

  equivalentSignals.unionSets(v1, v2);
  return true;
}
} // namespace llzk
