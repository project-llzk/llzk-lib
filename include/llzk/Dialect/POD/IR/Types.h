//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Dialect.h"

namespace llzk::pod {

struct RecordValue {
  mlir::StringRef name;
  mlir::Value value;
};

// Type alias for a list of pairs of (symbol, value) in the context of building `pod.new` ops.
using InitializedRecords = mlir::ArrayRef<RecordValue>;

} // namespace llzk::pod

// Include TableGen'd declarations
#define GET_TYPEDEF_CLASSES
#include "llzk/Dialect/POD/IR/Types.h.inc"

namespace llzk::pod {

mlir::ParseResult parsePodType(mlir::AsmParser &parser, mlir::SmallVector<RecordAttr> &);
void printPodType(mlir::AsmPrinter &printer, mlir::ArrayRef<RecordAttr>);

} // namespace llzk::pod
