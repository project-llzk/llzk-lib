//===-- Global.cpp - Global dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Ops.h"

#include "llzk-c/Dialect/Global.h"

#include <mlir/CAPI/Registration.h>

using namespace mlir;
using namespace llzk::global;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Global, llzk__global, GlobalDialect)

//===----------------------------------------------------------------------===//
// GlobalDefOp
//===----------------------------------------------------------------------===//

bool llzkOperationIsA_Global_GlobalDefOp(MlirOperation op) {
  return llvm::isa<GlobalDefOp>(unwrap(op));
}

bool llzkGlobal_GlobalDefOpGetIsConstant(MlirOperation op) {
  return unwrap_cast<GlobalDefOp>(op).isConstant();
}
