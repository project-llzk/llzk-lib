//===-- Ops.cpp - PCL dialect implementation ----------------*- C++ -*-----===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "pcl/Dialect/IR/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "pcl/Dialect/IR/Ops.cpp.inc"
