//===-- SharedImpl.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "SharedImpl.h"

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"

mlir::ConversionTarget llzk::polymorphic::detail::newBaseTarget(mlir::MLIRContext *ctx) {
  mlir::ConversionTarget target(*ctx);
  target.addLegalDialect<
      llzk::LLZKDialect, llzk::array::ArrayDialect, llzk::boolean::BoolDialect,
      llzk::cast::CastDialect, llzk::component::StructDialect, llzk::constrain::ConstrainDialect,
      llzk::felt::FeltDialect, llzk::function::FunctionDialect, llzk::global::GlobalDialect,
      llzk::include::IncludeDialect, llzk::polymorphic::PolymorphicDialect,
      llzk::string::StringDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  target.addLegalOp<mlir::ModuleOp>();
  return target;
}
