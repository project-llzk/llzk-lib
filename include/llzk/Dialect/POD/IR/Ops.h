//===-- Ops.h ---------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/Shared/DiscardableAllocationOpInterfaces.h"
#include "llzk/Dialect/Shared/OpHelpers.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/MemorySlotInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

// Include TableGen'd declarations
#include "llzk/Dialect/POD/IR/OpInterfaces.h.inc"

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/POD/IR/Ops.h.inc"

namespace llzk::pod {

mlir::SmallVector<RecordValue>
getInitializedRecordValues(mlir::ValueRange initialValues, mlir::ArrayAttr initializedRecords);

mlir::ParseResult parseRecordName(mlir::AsmParser &parser, mlir::StringAttr &name);
void printRecordName(mlir::AsmPrinter &printer, mlir::Operation *, mlir::StringAttr name);

} // namespace llzk::pod
