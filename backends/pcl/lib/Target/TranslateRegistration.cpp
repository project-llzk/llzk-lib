//===-- TranslateRegistration.cpp -------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "pcl/Target/TranslateRegistration.h"

#include "pcl/Dialect/IR/Dialect.h"
#include "pcl/Target/PCL.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Tools/mlir-translate/Translation.h>

using namespace mlir;

void pcl::registerPclTranslation() {
  mlir::TranslateFromMLIRRegistration reg(
      "pcl-to-lisp", "translate from pcl IR to pcl lisp",
      [](Operation *op, raw_ostream &output) -> LogicalResult {
    ModuleOp modOp = mlir::dyn_cast_if_present<ModuleOp>(op);
    if (!modOp) {
      return op->emitOpError() << "expected builtin.module as top level operation";
    }
    return pcl::moduleToPcl(modOp, output);
  }, [](DialectRegistry &registry) {
    registry.insert<
        // clang-format off
        pcl::PCLDialect,
        func::FuncDialect
        // clang-format on
        >();
  }
  );
}
