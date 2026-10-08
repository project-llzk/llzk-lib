//===-- LLZKLayoutTests.cpp - Logical signal layout tests -------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Util/LLZKLayout.h"

#include <mlir/IR/Builders.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace mlir;

class LLZKLayoutTests : public LLZKTest {};

TEST_F(LLZKLayoutTests, LookupUnusedSignalsByStructuralPath) {
  auto module = parseSourceString<ModuleOp>(
      R"(
    module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {
      struct.def @Main {
        struct.member @temporary : !felt.type
        struct.member @temporaryArray : !array.type<3 x !felt.type>
        struct.member @temporaryPod : !pod.type<[@x: !felt.type]>
        struct.member @z : !felt.type {llzk.pub}
        struct.member @a : !array.type<11 x !felt.type> {signal}
        struct.member @flag : i1
        function.def @compute(%input: !felt.type) -> !struct.type<@Main> {
          %self = struct.new : <@Main>
          function.return %self : !struct.type<@Main>
        }
        function.def @constrain(%self: !struct.type<@Main>, %input: !felt.type {function.arg_name = "named"}) {
          function.return
        }
      }
    }
  )",
      &ctx
  );
  ASSERT_TRUE(module);
  auto layout = llzk::buildLLZKLayout(*module);
  ASSERT_TRUE(succeeded(layout));
  ASSERT_EQ(layout->signalPaths.size(), 13);
  Builder builder(&ctx);
  auto main = builder.getStringAttr("main");
  auto member = builder.getStringAttr("a");
  for (unsigned index = 0; index < 11; ++index) {
    auto path = builder.getArrayAttr({main, member, builder.getIndexAttr(index)});
    EXPECT_EQ(layout->getSignalId(path), index + 2);
    EXPECT_EQ(layout->signalPaths[index + 2], path);
  }
  EXPECT_EQ(layout->getSignalId(builder.getArrayAttr({main, builder.getStringAttr("z")})), 1);
  EXPECT_EQ(
      layout->getSignalId(
          builder.getArrayAttr({builder.getStringAttr("arg"), builder.getI64IntegerAttr(1)})
      ),
      0
  );
  EXPECT_FALSE(layout->argumentNames[0]);
  EXPECT_EQ(layout->argumentNames[1].getValue(), "named");
  EXPECT_FALSE(layout->getSignalId(builder.getArrayAttr({main, builder.getStringAttr("flag")})));
  EXPECT_FALSE(
      layout->getSignalId(builder.getArrayAttr({main, builder.getStringAttr("temporary")}))
  );
  EXPECT_FALSE(layout->getSignalId(builder.getArrayAttr({main, member, builder.getIndexAttr(11)})));
  EXPECT_FALSE(layout->getSignalId({}));
  EXPECT_FALSE(layout->getSignalId(builder.getArrayAttr({builder.getStringAttr("named")})));
  EXPECT_FALSE(layout->getSignalId(builder.getArrayAttr({main, builder.getUnitAttr()})));
}

TEST_F(LLZKLayoutTests, IdsFollowMemberDeclarationOrder) {
  auto parse = [this](StringRef members) {
    std::string source = "module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {"
                         "struct.def @Main {" +
                         members.str() + R"(
      function.def @compute() -> !struct.type<@Main> {
        %self = struct.new : <@Main>
        function.return %self : !struct.type<@Main>
      }
      function.def @constrain(%self: !struct.type<@Main>) { function.return }
    }})";
    return parseSourceString<ModuleOp>(source, &ctx);
  };
  auto first =
      parse("struct.member @z : !felt.type {signal} struct.member @a : !felt.type {signal}");
  auto second =
      parse("struct.member @a : !felt.type {signal} struct.member @z : !felt.type {signal}");
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto firstLayout = llzk::buildLLZKLayout(*first);
  auto secondLayout = llzk::buildLLZKLayout(*second);
  ASSERT_TRUE(succeeded(firstLayout));
  ASSERT_TRUE(succeeded(secondLayout));
  Builder builder(&ctx);
  auto main = builder.getStringAttr("main");
  auto z = builder.getArrayAttr({main, builder.getStringAttr("z")});
  auto a = builder.getArrayAttr({main, builder.getStringAttr("a")});
  EXPECT_EQ(firstLayout->getSignalId(z), 0);
  EXPECT_EQ(firstLayout->getSignalId(a), 1);
  EXPECT_EQ(secondLayout->getSignalId(a), 0);
  EXPECT_EQ(secondLayout->getSignalId(z), 1);
}
