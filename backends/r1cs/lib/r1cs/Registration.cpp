//===-- Registration.cpp - Register R1CS dialect ---------------*- C++ -*--===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Dialect.h"
#include "r1cs/InitAllDialects.h"

#include <mlir/IR/DialectRegistry.h>

namespace r1cs {
void registerAllDialects(mlir::DialectRegistry &registry) { registry.insert<R1CSDialect>(); }
} // namespace r1cs
