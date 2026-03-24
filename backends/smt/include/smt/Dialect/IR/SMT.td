//===- SMT.td - SMT dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SMT_IR_SMT_TD
#define MLIR_DIALECT_SMT_IR_SMT_TD

include "mlir/IR/OpBase.td"

include "smt/Dialect/IR/SMTAttributes.td"
include "smt/Dialect/IR/SMTDialect.td"
include "smt/Dialect/IR/SMTTypes.td"
include "smt/Dialect/IR/SMTOps.td"
include "smt/Dialect/IR/SMTArrayOps.td"
include "smt/Dialect/IR/SMTBitVectorOps.td"
include "smt/Dialect/IR/SMTIntOps.td"

#endif // MLIR_DIALECT_SMT_IR_SMT_TD
