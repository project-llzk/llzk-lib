//===-- String.h - C API for String dialect -----------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// String dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_STRING_H
#define LLZK_C_DIALECT_STRING_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(String, llzk__string);

/// Creates a llzk::string::StringType.
MLIR_CAPI_EXPORTED MlirType llzkStringTypeGet(MlirContext);

/// Returns true if the type is a StringType.
LLZK_DECLARE_TYPE_ISA(String, StringType);

#ifdef __cplusplus
}
#endif

#endif
