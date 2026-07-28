//===-- PassBase.td - Base LLZK pass definition file -------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_PASS_PASSBASE
#define LLZK_PASS_PASSBASE

include "mlir/Pass/PassBase.td"

class LLZKPass<string passArg, string operation = "::mlir::ModuleOp">
    : Pass<passArg, operation> {
  let dependentDialects = [
      // clang-format off
      "llzk::LLZKDialect",
      "llzk::boolean::BoolDialect",
      "llzk::array::ArrayDialect",
      "llzk::component::StructDialect",
      "llzk::constrain::ConstrainDialect",
      "llzk::felt::FeltDialect",
      "llzk::global::GlobalDialect",
      "llzk::include::IncludeDialect",
      "llzk::function::FunctionDialect",
      "llzk::string::StringDialect",
      "llzk::polymorphic::PolymorphicDialect",
      "mlir::arith::ArithDialect",
      "mlir::scf::SCFDialect"
      // clang-format on
  ];
}

#endif // LLZK_PASS_PASSBASE
