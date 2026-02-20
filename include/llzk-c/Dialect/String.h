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

#include <mlir-c/IR.h>

// Include the generated CAPI
#include "llzk/Dialect/String/IR/Ops.capi.h.inc"
#include "llzk/Dialect/String/IR/Types.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(String, llzk__string);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECT_STRING_H
