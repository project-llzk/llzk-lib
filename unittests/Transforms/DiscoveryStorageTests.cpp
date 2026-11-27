//===-- DiscoveryStorageTests.cpp - Aggregate discovery bindings -----------===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../../lib/Dialect/Polymorphic/Transforms/StructSpecializationDiscovery.h"
#include "../LLZKTestBase.h"

#include <mlir/AsmParser/AsmParser.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;
using namespace llzk::polymorphic::detail;

namespace {

/// Exercise binding-aware discovery before monomorphization substitutes IR types.
class DiscoveryStorageTests : public LLZKTest {
protected:
  /// Inspect concrete child requests while the input IR still contains type variables.
  void checkBoundRead(llvm::StringRef source, llvm::StringRef boundType) {
    auto module = parseSourceString<ModuleOp>(source, &ctx);
    ASSERT_TRUE(module);
    auto parent = module->lookupSymbol<llzk::polymorphic::TemplateOp>("Parent");
    ASSERT_TRUE(parent);
    auto definition = *parent.getOps<llzk::component::StructDefOp>().begin();
    StructSpecializationDiscovery::Bindings bindings;
    bindings[FlatSymbolRefAttr::get(&ctx, "T")] = TypeAttr::get(parseType(boundType, &ctx));
    auto requests = StructSpecializationDiscovery().discover(definition, bindings);
    ASSERT_TRUE(succeeded(requests));
    bool found = false;
    for (const auto &request : *requests) {
      if (request.type.getNameRef().getLeafReference().getValue() == "Child") {
        ASSERT_EQ(request.type.getParams().size(), 1U);
        auto argument = dyn_cast<IntegerAttr>(request.type.getParams()[0]);
        ASSERT_TRUE(argument);
        EXPECT_EQ(argument.getInt(), 2);
        found = true;
      }
    }
    EXPECT_TRUE(found);
  }
};

TEST_F(DiscoveryStorageTests, BoundPodReadSnapshotsContents) {
  checkBoundRead(
      R"llzk(
module attributes {llzk.lang, llzk.main = !struct.type<@Parent::@Parent<[!pod.type<[@n: index]>]>>} {
  poly.template @Child {
    poly.param @N : index
    struct.def @Child {
      function.def @compute() -> !struct.type<@Child::@Child<[@N]>> {
        %s = struct.new : !struct.type<@Child::@Child<[@N]>>
        function.return %s : !struct.type<@Child::@Child<[@N]>>
      }
      function.def @constrain(%s: !struct.type<@Child::@Child<[@N]>>) {
        function.return
      }
    }
  }
  poly.template @Parent {
    poly.param @T : !poly.tvar<@T>
    struct.def @Parent {
      function.def @compute(%input: !felt.type) -> !struct.type<@Parent::@Parent<[@T]>> {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %leaf = pod.new { @n = %two } : !pod.type<[@n: index]>
        %holder = pod.new : !pod.type<[@item: !poly.tvar<@T>]>
        %erased = poly.unifiable_cast %leaf : (!pod.type<[@n: index]>) -> !poly.tvar<@T>
        pod.write %holder[@item] = %erased : !pod.type<[@item: !poly.tvar<@T>]>, !poly.tvar<@T>
        %copy = pod.read %holder[@item] : !pod.type<[@item: !poly.tvar<@T>]>, !poly.tvar<@T>
        %typed = poly.unifiable_cast %copy : (!poly.tvar<@T>) -> !pod.type<[@n: index]>
        %value = pod.read %typed[@n] : !pod.type<[@n: index]>, index
        %child = function.call @Child::@Child::@compute(){(%value)} : () -> !struct.type<@Child::@Child<[affine_map<(i) -> (i)>]>>
        %self = struct.new : !struct.type<@Parent::@Parent<[@T]>>
        function.return %self : !struct.type<@Parent::@Parent<[@T]>>
      }
      function.def @constrain(%self: !struct.type<@Parent::@Parent<[@T]>>, %input: !felt.type) {
        function.return
      }
    }
  }
}
)llzk",
      "!pod.type<[@n: index]>"
  );
}

TEST_F(DiscoveryStorageTests, BoundArrayElementSnapshotsContents) {
  checkBoundRead(
      R"llzk(
module attributes {llzk.lang, llzk.main = !struct.type<@Parent::@Parent<[!array.type<1 x index>]>>} {
  poly.template @Child {
    poly.param @N : index
    struct.def @Child {
      function.def @compute() -> !struct.type<@Child::@Child<[@N]>> {
        %s = struct.new : !struct.type<@Child::@Child<[@N]>>
        function.return %s : !struct.type<@Child::@Child<[@N]>>
      }
      function.def @constrain(%s: !struct.type<@Child::@Child<[@N]>>) {
        function.return
      }
    }
  }
  poly.template @Parent {
    poly.param @T : !poly.tvar<@T>
    struct.def @Parent {
      function.def @compute(%input: !felt.type) -> !struct.type<@Parent::@Parent<[@T]>> {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %leaf = array.new %two : !array.type<1 x index>
        %erased = poly.unifiable_cast %leaf : (!array.type<1 x index>) -> !poly.tvar<@T>
        %slots = array.new %erased : !array.type<1 x !poly.tvar<@T>>
        %copy = array.read %slots[%zero] : !array.type<1 x !poly.tvar<@T>>, !poly.tvar<@T>
        %typed = poly.unifiable_cast %copy : (!poly.tvar<@T>) -> !array.type<1 x index>
        %value = array.read %typed[%zero] : !array.type<1 x index>, index
        %child = function.call @Child::@Child::@compute(){(%value)} : () -> !struct.type<@Child::@Child<[affine_map<(i) -> (i)>]>>
        %self = struct.new : !struct.type<@Parent::@Parent<[@T]>>
        function.return %self : !struct.type<@Parent::@Parent<[@T]>>
      }
      function.def @constrain(%self: !struct.type<@Parent::@Parent<[@T]>>, %input: !felt.type) {
        function.return
      }
    }
  }
}
)llzk",
      "!array.type<1 x index>"
  );
}

} // namespace
