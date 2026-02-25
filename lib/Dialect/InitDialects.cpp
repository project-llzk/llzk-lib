//===-- InitDialects.cpp - LLZK dialect registration ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Include/IR/Dialect.h"
#include "llzk/Dialect/InitDialects.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/POD/IR/Dialect.h"
#include "llzk/Dialect/Polymorphic/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>

namespace llzk {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      llzk::LLZKDialect,
      llzk::array::ArrayDialect,
      llzk::boolean::BoolDialect,
      llzk::cast::CastDialect,
      llzk::component::StructDialect,
      llzk::constrain::ConstrainDialect,
      llzk::felt::FeltDialect,
      llzk::function::FunctionDialect,
      llzk::global::GlobalDialect,
      llzk::include::IncludeDialect,
      llzk::string::StringDialect,
      llzk::pod::PODDialect,
      llzk::polymorphic::PolymorphicDialect,
      mlir::arith::ArithDialect,
      mlir::scf::SCFDialect
      // clang-format on
      >();

  registerInliningExtensions(registry);
}
} // namespace llzk
