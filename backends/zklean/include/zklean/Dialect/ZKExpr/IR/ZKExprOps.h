//===-- ZKExprOps.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "zklean/Dialect/ZKExpr/IR/ZKExprDialect.h"
#include "zklean/Dialect/ZKExpr/IR/ZKExprTypes.h"
#include "llzk/Dialect/Felt/IR/Types.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#define GET_OP_CLASSES
#include "zklean/Dialect/ZKExpr/IR/ZKExprOps.h.inc"
