//===-- Include.h - C API for Include dialect ---------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Include dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_INCLUDE_H
#define LLZK_C_DIALECT_INCLUDE_H

#include "llzk/Dialect/Include/Transforms/InlineIncludesPass.capi.h.inc"

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Include, llzk__include);

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

/// Creates an IncludeOp pointing to another MLIR file.
MLIR_CAPI_EXPORTED MlirOperation
llzkInclude_IncludeOpCreate(MlirLocation loc, MlirStringRef name, MlirStringRef path);

#ifdef __cplusplus
}
#endif

#endif
