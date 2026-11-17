//===-- TemplateInstantiationTests.cpp - Expression evaluation --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../../lib/Dialect/Polymorphic/Transforms/TemplateInstantiation.h"
#include "../LLZKTestBase.h"

#include <mlir/Parser/Parser.h>

#include <llvm/Support/raw_ostream.h>

#include <string>

using namespace mlir;
using namespace llzk::polymorphic;

namespace {

class TemplateInstantiationTests : public LLZKTest {};

// Inspect the evaluator directly: the full pass can independently canonicalize
// retained template bodies, which would hide mutations performed by evaluation.
TEST_F(TemplateInstantiationTests, ExpressionEvaluationPreservesSharedBody) {
  auto module = parseSourceString<ModuleOp>(
      R"mlir(
    module attributes {llzk.lang} {
      poly.template @T {
        poly.param @N : index
        poly.expr @Extended {
          %n = poly.read_const @N : index
          %byte = arith.index_castui %n : index to i8
          %half = arith.extui %byte : i8 to i16
          %word = arith.extui %half : i16 to i32
          %result = arith.index_castui %word : i32 to index
          poly.yield %result : index
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);
  TemplateOp templ = *module->getOps<TemplateOp>().begin();
  auto printTemplate = [&]() {
    std::string text;
    llvm::raw_string_ostream os(text);
    templ.print(os);
    return text;
  };
  std::string original = printTemplate();
  OpBuilder builder(&ctx);
  auto param = FlatSymbolRefAttr::get(&ctx, "N");
  auto expr = FlatSymbolRefAttr::get(&ctx, "Extended");
  for (int64_t value : {1, 2}) {
    llvm::DenseMap<Attribute, Attribute> bindings {{param, builder.getIndexAttr(value)}};
    llzk::polymorphic::detail::evaluateTemplateExprs(templ, bindings);
    EXPECT_EQ(bindings.lookup(expr), builder.getIndexAttr(value));
    EXPECT_EQ(printTemplate(), original);
  }

  // Unresolved evaluation must also preserve the original expression and bindings.
  llvm::DenseMap<Attribute, Attribute> unresolved;
  llzk::polymorphic::detail::evaluateTemplateExprs(templ, unresolved);
  EXPECT_TRUE(unresolved.empty());
  EXPECT_EQ(printTemplate(), original);
}

} // namespace
