//===-- SMTLIBEmitter.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/Support/raw_ostream.h>

namespace llzk::smt {

mlir::LogicalResult emitSMTLIBModule(mlir::ModuleOp module, llvm::raw_ostream &os);

} // namespace llzk::smt
