//===-- LLZK.cpp - LLZK dialect C API implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/LLZK/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"

#include "llzk-c/Dialect/LLZK.h"

#include <mlir/CAPI/Registration.h>

using namespace llzk;

// Include the generated CAPI
#include "llzk/Dialect/LLZK/IR/Attrs.capi.cpp.inc"
#include "llzk/Dialect/LLZK/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLZK, llzk, llzk::LLZKDialect)
