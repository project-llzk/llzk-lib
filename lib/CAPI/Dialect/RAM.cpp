//===-- RAM.cpp - RAM dialect C API implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/RAM.h"

#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/RAM/IR/Dialect.h"
#include "llzk/Dialect/RAM/IR/Ops.h"

#include <mlir/CAPI/Registration.h>

using namespace llzk::ram;

// Include the generated CAPI
#include "llzk/Dialect/RAM/IR/Ops.capi.cpp.inc"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(RAM, llzk__ram, RAMDialect)
