//===-- Global.h - C API for Global dialect -----------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Global dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_GLOBAL_H
#define LLZK_C_DIALECT_GLOBAL_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Global, llzk__global);

//===----------------------------------------------------------------------===//
// GlobalDefOp
//===----------------------------------------------------------------------===//

/// Returns true if the op is a GlobalDefOp.
LLZK_DECLARE_OP_ISA(Global, GlobalDefOp);

/// Returns true if the op defines a constant value.
LLZK_DECLARE_OP_PREDICATE(Global, GlobalDefOp, IsConstant);

#ifdef __cplusplus
}
#endif

#endif
