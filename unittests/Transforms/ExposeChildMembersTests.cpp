//===-- ExposeChildMembersTests.cpp ---------------------------------------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Struct/Transforms/ExposeChildMembersPass.h"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk::component;

namespace {
class ExposeChildMembersTests : public LLZKTest {};

TEST_F(ExposeChildMembersTests, FailureDoesNotExposePreviouslyCollectedChildren) {
  constexpr llvm::StringLiteral source = R"mlir(
    module attributes {llzk.lang} {
      struct.def @Child {
        struct.member @value : !felt.type {signal}
        function.def @product() -> !struct.type<@Child> {
          %self = struct.new : <@Child>
          function.return %self : !struct.type<@Child>
        }
      }
      poly.template @T {
        poly.param @N : index
        struct.def @T {
          function.def @product() -> !struct.type<@T::@T<[@N]>> {
            %self = struct.new : <@T::@T<[@N]>>
            function.return %self : !struct.type<@T::@T<[@N]>>
          }
        }
      }
      struct.def @Root {
        struct.member @child : !struct.type<@Child>
        struct.member @unresolved : !array.type<1 x !struct.type<@T::@T<[affine_map<(i) -> (i)>]>>>
        function.def @product() -> !struct.type<@Root> {
          %self = struct.new : <@Root>
          function.return %self : !struct.type<@Root>
        }
      }
    }
  )mlir";
  auto module = parseSourceString<ModuleOp>(source, &ctx);
  ASSERT_TRUE(module);
  auto root = module->lookupSymbol<StructDefOp>("Root");
  auto child = module->lookupSymbol<StructDefOp>("Child");
  ASSERT_TRUE(root);
  ASSERT_TRUE(child);

  unsigned errors = 0;
  ScopedDiagnosticHandler handler(&ctx, [&](Diagnostic &diagnostic) {
    EXPECT_EQ(
        diagnostic.str(), "missing concrete struct family element for llzk-expose-child-members"
    );
    ++errors;
    return success();
  });
  EXPECT_TRUE(failed(exposeChildMembers(root)));
  EXPECT_EQ(errors, 1U);
  EXPECT_FALSE(child.getMemberDefs().front().hasPublicAttr());
  for (auto member : root.getMemberDefs()) {
    EXPECT_FALSE(member.hasPublicAttr());
  }
}
} // namespace
