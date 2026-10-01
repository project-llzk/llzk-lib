//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Defines dialect and pass registration functions for core LLZK dialects.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/DialectRegistration.h"

#include "llzk-c/DialectRegistration.h"

#include <mlir-c/IR.h>

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Wrap.h>

void llzkRegisterAllDialects(MlirDialectRegistry registry) {
  llzk::registerDialects(*unwrap(registry));
}
