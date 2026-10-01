//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/DialectRegistration.h"

#include <mlir-c/IR.h>

#include <gtest/gtest.h>

TEST(DialectRegistration, RegisterDialects) {
  MlirDialectRegistry registry = mlirDialectRegistryCreate();
  llzkRegisterCoreDialects(registry);
  mlirDialectRegistryDestroy(registry);
}
