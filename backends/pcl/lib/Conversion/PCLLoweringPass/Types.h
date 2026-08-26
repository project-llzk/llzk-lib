//===-- Modes.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "pcl/Dialect/IR/Ops.h"

#include "llzk/Dialect/LLZK/IR/Ops.h"

namespace pcl::lowering {
using NonDetOpNames = llvm::DenseMap<llzk::NonDetOp, mlir::StringAttr>;
using DupVarsReplacements = llvm::DenseMap<pcl::VarOp, mlir::Value>;
} // namespace pcl::lowering
