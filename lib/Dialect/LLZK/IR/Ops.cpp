//===-- Ops.cpp - LLZK operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Util/TypeHelper.h"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/LLZK/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk {

//===------------------------------------------------------------------===//
// NonDetOp
//===------------------------------------------------------------------===//

void NonDetOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "nondet");
}

} // namespace llzk
