//===-- Attrs.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>

// Include TableGen'd declarations
#define GET_ATTRDEF_CLASSES
#include "llzk/Dialect/POD/IR/Attrs.h.inc"

namespace llzk::pod {

mlir::ParseResult parseRecord(mlir::AsmParser &parser, mlir::StringAttr &name, mlir::Type &type);
void printRecord(mlir::AsmPrinter &printer, mlir::StringAttr name, mlir::Type type);

} // namespace llzk::pod
