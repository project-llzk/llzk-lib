//===-- Support.h - C API general utilities -----------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares utilities for working with the C API.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_IR_H
#define LLZK_C_IR_H

#include "llzk-c/Builder.h" // IWYU pragma: keep

#include <mlir-c/IR.h> // IWYU pragma: keep

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Utility macros for function declarations.
//===----------------------------------------------------------------------===//

#define LLZK_BUILD_METHOD_NAME(dialect, op, suffix) llzk##dialect##_##op##Build##suffix
#define LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(dialect, op, suffix, ...)                              \
  MLIR_CAPI_EXPORTED MlirOperation LLZK_BUILD_METHOD_NAME(dialect, op, suffix)(                    \
      MlirOpBuilder builder, MlirLocation location, __VA_ARGS__                                    \
  )
// Used for when the build method is "general" and does not have a suffix at the end.
#define LLZK_DECLARE_OP_BUILD_METHOD(dialect, op, ...)                                             \
  LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(dialect, op, , __VA_ARGS__)

#define LLZK_DECLARE_OP_PREDICATE(dialect, op, name)                                               \
  MLIR_CAPI_EXPORTED bool llzk##dialect##_##op##Get##name(MlirOperation op)
#define LLZK_DECLARE_NARY_OP_PREDICATE(dialect, op, name, ...)                                     \
  MLIR_CAPI_EXPORTED bool llzk##dialect##_##op##Get##name(MlirOperation op, __VA_ARGS__)

//===----------------------------------------------------------------------===//
// Representation of a mlir::ValueRange.
//===----------------------------------------------------------------------===//

struct MlirValueRange {
  MlirValue const *values;
  intptr_t size;
};
typedef struct MlirValueRange MlirValueRange;

#ifdef __cplusplus
}
#endif

#endif
