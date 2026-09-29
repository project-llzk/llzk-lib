//===-- SMTInfoOps.h - SMT metadata operations ------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/SMTInfo/IR/SMTInfoAttributes.h"
#include "llzk/Dialect/SMTInfo/IR/SMTInfoDialect.h"

#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>

#define GET_OP_CLASSES
#include "llzk/Dialect/SMTInfo/IR/SMTInfoOps.h.inc"
