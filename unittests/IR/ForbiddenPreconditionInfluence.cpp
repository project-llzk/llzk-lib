//===-- ForbiddenPrconditionInfluence.cpp -----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/Util/ForbiddenPreconditionInfluence.h"

#include "../LLZKTestBase.h"
#include "../LLZKTestUtils.h"

#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Parser/Parser.h>

#include <algorithm>
#include <gtest/gtest.h>
#include <string>

using namespace mlir;
using namespace llzk;
using namespace llzk::component;
using namespace llzk::verif;

class ForbiddenPreconditionInfluenceTests : public LLZKTest {};

TEST_F(ForbiddenPreconditionInfluenceTests, TracksHelperReturnThroughInternalIfControl) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  function.def @helper(%secret: !felt.type, %fallback: !felt.type, %gate: i1) -> !felt.type {
    %selected = scf.if %gate -> (!felt.type) {
      scf.yield %secret : !felt.type
    } else {
      scf.yield %fallback : !felt.type
    }
    function.return %selected : !felt.type
  }
}
)mlir";

  MLIRContext localCtx;
  DialectRegistry registry;
  llzk::registerAllDialects(registry);
  localCtx.appendDialectRegistry(registry);
  localCtx.loadAllAvailableDialects();

  auto parsed = parseSourceString<ModuleOp>(source, ParserConfig(&localCtx));
  ASSERT_TRUE(parsed);

  function::FuncDefOp helper;
  parsed->walk([&](function::FuncDefOp op) {
    if (op.getName() == "helper") {
      helper = op;
    }
  });
  ASSERT_TRUE(helper);

  SmallVector<ForbiddenPreconditionInfluence> argInfluences {
      ForbiddenPreconditionInfluence::StructMember,
      ForbiddenPreconditionInfluence::None,
      ForbiddenPreconditionInfluence::StructMember,
  };
  auto influence =
      analyzeForbiddenPreconditionCallableResult(parsed.get(), helper, argInfluences, 0);
  EXPECT_TRUE(hasInfluence(influence, ForbiddenPreconditionInfluence::StructMember));
}

TEST_F(ForbiddenPreconditionInfluenceTests, TracksTransitiveHelperReturnControl) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  function.def @inner(%ret_value: i1, %fallback: i1) -> i1 {
    %true = arith.constant true
    %false = arith.constant false
    %selected = scf.if %ret_value -> (i1) {
      scf.yield %true : i1
    } else {
      scf.yield %false : i1
    }
    %result = arith.ori %selected, %fallback : i1
    function.return %result : i1
  }

  function.def @outer(%ret_value: i1, %fallback: i1) -> i1 {
    %result = function.call @inner(%ret_value, %fallback) : (i1, i1) -> i1
    function.return %result : i1
  }
}
)mlir";

  MLIRContext localCtx;
  DialectRegistry registry;
  llzk::registerAllDialects(registry);
  localCtx.appendDialectRegistry(registry);
  localCtx.loadAllAvailableDialects();

  auto parsed = parseSourceString<ModuleOp>(source, ParserConfig(&localCtx));
  ASSERT_TRUE(parsed);

  function::FuncDefOp outer;
  parsed->walk([&](function::FuncDefOp op) {
    if (op.getName() == "outer") {
      outer = op;
    }
  });
  ASSERT_TRUE(outer);

  SmallVector<ForbiddenPreconditionInfluence> argInfluences {
      ForbiddenPreconditionInfluence::FunctionReturn,
      ForbiddenPreconditionInfluence::None,
  };
  auto influence =
      analyzeForbiddenPreconditionCallableResult(parsed.get(), outer, argInfluences, 0);
  EXPECT_TRUE(hasInfluence(influence, ForbiddenPreconditionInfluence::FunctionReturn));
}
