//===-- TranslateRegistration.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "smt/Target/TranslateRegistration.h"

#include "smt/Target/SMTLIBEmitter.h"

#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SMT/IR/SMTDialect.h>
#include <mlir/Tools/mlir-translate/Translation.h>

using namespace mlir;
using namespace llzk;

void llzk::smt::registerSmtTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "smt-to-smtlib", "translate from SMT to SMTLIB",
      [](Operation *op, raw_ostream &output) -> LogicalResult {
    ModuleOp modOp = llvm::dyn_cast_if_present<ModuleOp>(op);
    if (!modOp) {
      return op->emitOpError() << "expected builtin.module as top level operation";
    }
    return llzk::smt::emitSMTLIBModule(modOp, output);
  }, [](DialectRegistry &registry) {
    registry.insert<
        // clang-format off
        func::FuncDialect,
        mlir::smt::SMTDialect,
        smt_info::SMTInfoDialect,
        boolean::BoolDialect,
        arith::ArithDialect
        // clang-format on
        >();
  }
  );
}
