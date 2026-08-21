//===-- Dialect.cpp - Channel dialect implementation ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Channel/IR/Dialect.h"

#include "llzk/Dialect/Channel/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

// TableGen'd implementation files
#include "llzk/Dialect/Channel/IR/Dialect.cpp.inc"

//===------------------------------------------------------------------===//
// ChannelDialect
//===------------------------------------------------------------------===//

auto llzk::channel::ChannelDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Channel/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<ChannelDialect>>();
}
