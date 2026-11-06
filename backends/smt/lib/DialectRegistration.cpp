//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
/// \file
/// Defines dialect and pass registration functions for SMT backend.
//
//===----------------------------------------------------------------------===//

#include "smt/DialectRegistration.h"

#include "smt/Conversions/ConversionPasses.h"

#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.h"

#include <mlir/Dialect/Func/Extensions/InlinerExtension.h>
#include <mlir/Dialect/SMT/IR/SMTDialect.h>
#include <mlir/IR/DialectRegistry.h>

namespace llzk::smt {

void registerDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::smt::SMTDialect, llzk::smt_info::SMTInfoDialect>();
}

void registerPasses(mlir::DialectRegistry &registry) {
  registerConversionPasses();
  mlir::func::registerInlinerExtension(registry);
}

} // namespace llzk::smt
