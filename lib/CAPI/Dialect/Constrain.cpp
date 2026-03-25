//===-- Constrain.cpp - Constrain dialect C API implementation --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/Constrain/IR/Dialect.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"

#include "llzk-c/Dialect/Constrain.h"

#include <mlir/CAPI/Registration.h>

using namespace llzk::constrain;

// Include the generated CAPI
#include "llzk/Dialect/Constrain/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Constrain, llzk__constrain, ConstrainDialect)
