//===-- Attrs.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/DialectImplementation.h>

// need to include APSInt.h for Attrs.h.inc
#include <llvm/ADT/APSInt.h>
// Include TableGen'd declarations
#define GET_ATTRDEF_CLASSES
#include "r1cs/Dialect/IR/Attrs.h.inc"
