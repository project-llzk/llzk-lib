//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "pcl/Dialect/IR/Dialect.h"
#include "pcl/Dialect/IR/Types.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

// Include TableGen'd op interfaces (if defined)
#ifdef GET_OP_INTERFACE_DECLS
#include "pcl/Dialect/IR/OpInterfaces.h.inc"
#endif

// Include TableGen'd op classes
#define GET_OP_CLASSES
#include "pcl/Dialect/IR/Ops.h.inc"
