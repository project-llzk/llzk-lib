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

include "llzk/Dialect/SMT/IR/SMTAttributes.td"
include "llzk/Dialect/SMT/IR/SMTDialect.td"
include "llzk/Dialect/SMT/IR/SMTTypes.td"
include "llzk/Dialect/SMT/IR/SMTOps.td"
include "llzk/Dialect/SMT/IR/SMTArrayOps.td"
include "llzk/Dialect/SMT/IR/SMTBitVectorOps.td"
include "llzk/Dialect/SMT/IR/SMTIntOps.td"

#endif // MLIR_DIALECT_SMT_IR_SMT_TD
