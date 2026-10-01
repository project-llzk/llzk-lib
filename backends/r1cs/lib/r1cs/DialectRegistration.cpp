//===-- DialectRegistration.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines dialect and pass registration functions for R1CS backend.
//
//===----------------------------------------------------------------------===//

#include "r1cs/DialectRegistration.h"

#include "r1cs/Dialect/IR/Dialect.h"

#include <mlir/IR/DialectRegistry.h>

namespace r1cs {

void registerDialects(mlir::DialectRegistry &registry) { registry.insert<R1CSDialect>(); }

} // namespace r1cs
