//===-- Ops.cpp - Verif operation implementations ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/IR/Ops.h"

#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#include "llzk/Dialect/Verif/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Verif/IR/Ops.cpp.inc"

using namespace mlir;

namespace llzk::verif {

//===------------------------------------------------------------------===//
// ContractOp
//===------------------------------------------------------------------===//

LogicalResult ContractOp::verify() { return emitOpError() << "verification not yet implemented"; }

LogicalResult ContractOp::verifySymbolUses(SymbolTableCollection &tables) {
  return emitOpError() << "verifySymbolUses not yet implemented";
}

ParseResult ContractOp::parse(OpAsmParser &parser, OperationState &result) { return failure(); }

void ContractOp::print(OpAsmPrinter &p) {}

} // namespace llzk::verif
