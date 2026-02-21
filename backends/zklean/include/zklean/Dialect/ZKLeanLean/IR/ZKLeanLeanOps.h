//===-- ZKLeanLeanOps.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "zklean/Dialect/ZKExpr/IR/ZKExprTypes.h"
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanDialect.h"
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanTypes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

#define GET_OP_CLASSES
#include "zklean/Dialect/ZKLeanLean/IR/ZKLeanLeanOps.h.inc"
