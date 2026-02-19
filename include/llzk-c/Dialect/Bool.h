//===-- Bool.h - C API for Bool dialect ---------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Bool dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_BOOL_H
#define LLZK_C_DIALECT_BOOL_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Bool, llzk__boolean);

enum LlzkCmp {
  LlzkCmp_EQ = 0,
  LlzkCmp_NE = 1,
  LlzkCmp_LT = 2,
  LlzkCmp_LE = 3,
  LlzkCmp_GT = 4,
  LlzkCmp_GE = 5
};
typedef enum LlzkCmp LlzkCmp;

/// Returns a llzk::boolean::FeltCmpPredicateAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute llzkBool_FeltCmpPredicateAttrGet(MlirContext context, LlzkCmp cmp);

/// Returns true if the attribute is a FeltCmpPredicateAttr.
LLZK_DECLARE_ATTR_ISA(Bool, FeltCmpPredicateAttr);

#ifdef __cplusplus
}
#endif

#endif
