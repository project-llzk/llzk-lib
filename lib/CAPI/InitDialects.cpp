//===-- InitDialects.cpp - C API for dialect registration -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/InitDialects.h"

#include "llzk-c/InitDialects.h"

#include <mlir-c/IR.h>

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Wrap.h>

void llzkRegisterAllDialects(MlirDialectRegistry registry) {
  llzk::registerAllDialects(*unwrap(registry));
}
