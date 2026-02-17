//===-- ZKBuilderOps.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderDialect.h"
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderTypes.h"
#include "llzk/Dialect/ZKExpr/IR/ZKExprTypes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/Interfaces/InferTypeOpInterface.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/ZKBuilder/IR/ZKBuilderOps.h.inc"
