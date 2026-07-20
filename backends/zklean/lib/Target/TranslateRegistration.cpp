//===-- TranslateRegistration.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "zklean/Target/TranslateRegistration.h"

#include "zklean/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "zklean/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"
#include "zklean/Target/ZKLean.h"

#include "llzk/Dialect/Felt/IR/Dialect.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Tools/mlir-translate/Translation.h>

using namespace mlir;
using namespace llzk;

void zklean::registerZKLeanTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "zklean-to-lean", "pretty-print zkLean dialect IR as Lean code",
      [](Operation *op, raw_ostream &output) -> LogicalResult {
    ModuleOp modOp = mlir::dyn_cast_if_present<ModuleOp>(op);
    if (!modOp) {
      return op->emitOpError() << "expected builtin.module as top level operation";
    }
    return zklean::emitZKLeanModule(modOp, output);
  }, [](DialectRegistry &registry) {
    registry.insert<
        // clang-format off
        felt::FeltDialect,
        func::FuncDialect,
        zkexpr::ZKExprDialect,
        zkbuilder::ZKBuilderDialect,
        zkleanlean::ZKLeanLeanDialect
        // clang-format on
        >();
  }
  );
}
