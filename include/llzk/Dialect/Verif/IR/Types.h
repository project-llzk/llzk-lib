//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Util/ErrorHelper.h"

#include <mlir/Interfaces/MemorySlotInterfaces.h>

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/Array/IR/Types.h.inc"

namespace llzk::array {

mlir::LogicalResult computeDimsFromShape(
    mlir::MLIRContext *ctx, llvm::ArrayRef<int64_t> shape,
    llvm::SmallVector<mlir::Attribute> &dimensionSizes
);

mlir::LogicalResult computeShapeFromDims(
    EmitErrorFn emitError, llvm::ArrayRef<mlir::Attribute> dimensionSizes,
    llvm::SmallVector<int64_t> &shape
);

mlir::ParseResult parseDerivedShape(
    mlir::AsmParser &parser, llvm::SmallVector<int64_t> &shape,
    llvm::SmallVector<mlir::Attribute> dimensionSizes
);
void printDerivedShape(
    mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<mlir::Attribute> dimensionSizes
);

mlir::ParseResult parseAttrVec(mlir::AsmParser &parser, llvm::SmallVector<mlir::Attribute> &value);
void printAttrVec(mlir::AsmPrinter &printer, llvm::ArrayRef<mlir::Attribute> value);

} // namespace llzk::array
