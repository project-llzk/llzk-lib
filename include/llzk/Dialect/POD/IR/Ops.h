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
#include "llzk/Dialect/Shared/OpHelpers.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>

// Include TableGen'd declarations
#define GET_OP_CLASSES
#include "llzk/Dialect/POD/IR/Ops.h.inc"

namespace llzk::pod {

mlir::ParseResult parseRecordName(mlir::AsmParser &parser, mlir::FlatSymbolRefAttr &name);
void printRecordName(mlir::AsmPrinter &printer, mlir::Operation *, mlir::FlatSymbolRefAttr name);

} // namespace llzk::pod
