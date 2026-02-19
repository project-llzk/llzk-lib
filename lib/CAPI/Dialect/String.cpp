//===-- String.cpp - String dialect C API implementation --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/String/IR/Dialect.h"
#include "llzk/Dialect/String/IR/Types.h"

#include "llzk-c/Dialect/String.h"

#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Wrap.h>

using namespace mlir;
using namespace llzk::string;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(String, llzk__string, StringDialect)

MlirType llzkStringTypeGet(MlirContext ctx) { return wrap(StringType::get(unwrap(ctx))); }

bool llzkTypeIsA_String_StringType(MlirType type) { return llvm::isa<StringType>(unwrap(type)); }
