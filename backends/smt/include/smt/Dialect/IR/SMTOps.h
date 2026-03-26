//===- SMTOps.h - SMT dialect operations ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMTOPS_H
#define MLIR_DIALECT_SMT_IR_SMTOPS_H

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "smt/Dialect/IR/SMTAttributes.h"
#include "smt/Dialect/IR/SMTDialect.h"
#include "smt/Dialect/IR/SMTTypes.h"

#define GET_OP_CLASSES
#include "smt/Dialect/IR/SMT.h.inc"

#endif // MLIR_DIALECT_SMT_IR_SMTOPS_H
