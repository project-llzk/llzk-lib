//===-- SMTLoweringPass.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-smt-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Dialect.h"
#include "llzk/Dialect/Bool/IR/Dialect.h"
#include "llzk/Dialect/Cast/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Felt/IR/Dialect.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/Struct/IR/Dialect.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>

#include "smt/Conversions/ConversionPasses.h"

namespace smt {
#define GEN_PASS_DEF_SMTLOWERINGPASS
#include "smt/Conversions/ConversionPasses.h.inc"
} // namespace smt
