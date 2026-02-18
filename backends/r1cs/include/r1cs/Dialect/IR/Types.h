//===-- Types.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once
#include "r1cs/Dialect/IR/Dialect.h"

#include <llvm/ADT/TypeSwitch.h>

#define GET_TYPEDEF_CLASSES
#include "r1cs/Dialect/IR/Types.h.inc"
