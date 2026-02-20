//===-- String.cpp - String dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Ops.h"
#include "llzk/Dialect/String/IR/Types.h"

#include "llzk-c/Dialect/String.h"

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace mlir;
using namespace llzk::string;

// Include the generated CAPI
#include "llzk/Dialect/String/IR/Ops.capi.cpp.inc"
#include "llzk/Dialect/String/IR/Types.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(String, llzk__string, StringDialect)
