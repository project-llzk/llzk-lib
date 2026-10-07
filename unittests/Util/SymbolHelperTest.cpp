//===-- SymbolHelperTest.cpp - Unit tests for symbol utilities --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/SymbolHelper.h"

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Debug.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>
#include <iterator>

using namespace llzk;
using namespace mlir;

class SymbolHelperTests : public LLZKTest {
protected:
  SymbolHelperTests() : LLZKTest() {}

  SymbolRefAttr newExample(unsigned numNestedRefs = 0) {
    llvm::SmallVector<FlatSymbolRefAttr> nestedRefs;
    for (unsigned i = 0; i < numNestedRefs; i++) {
      nestedRefs.push_back(FlatSymbolRefAttr::get(&ctx, StringAttr::get(&ctx, "r" + Twine(i + 1))));
    }
    return SymbolRefAttr::get(&ctx, "root", nestedRefs);
  }
};

TEST_F(SymbolHelperTests, test_getFlatSymbolRefAttr) {
  FlatSymbolRefAttr attr = getFlatSymbolRefAttr(&ctx, "name");
  ASSERT_EQ(attr.getValue(), "name");
}

TEST_F(SymbolHelperTests, test_getNames) {
  SymbolRefAttr attr = newExample(3);
  ASSERT_EQ(debug::toStringOne(attr), "@root::@r1::@r2::@r3");

  llvm::SmallVector<StringRef> names = getNames(attr);
  ASSERT_EQ(names.size(), 4);
  ASSERT_EQ(names, SmallVector<StringRef>({"root", "r1", "r2", "r3"}));
}

TEST_F(SymbolHelperTests, test_getPieces) {
  SymbolRefAttr attr = newExample(3);
  ASSERT_EQ(debug::toStringOne(attr), "@root::@r1::@r2::@r3");

  llvm::SmallVector<FlatSymbolRefAttr> pieces = getPieces(attr);
  ASSERT_EQ(pieces.size(), 4);
  ASSERT_EQ(
      pieces, SmallVector<FlatSymbolRefAttr>(
                  {FlatSymbolRefAttr::get(&ctx, "root"), FlatSymbolRefAttr::get(&ctx, "r1"),
                   FlatSymbolRefAttr::get(&ctx, "r2"), FlatSymbolRefAttr::get(&ctx, "r3")}
              )
  );
}

TEST_F(SymbolHelperTests, test_asSymbolRefAttr_StringAttr_SymRefAttr) {
  SymbolRefAttr attr = asSymbolRefAttr(StringAttr::get(&ctx, "super"), newExample(2));
  ASSERT_EQ(debug::toStringOne(attr), "@super::@root::@r1::@r2");
}

TEST_F(SymbolHelperTests, test_asSymbolRefAttr_ArrRef_Flat) {
  SymbolRefAttr attr = asSymbolRefAttr(ArrayRef(
      {FlatSymbolRefAttr::get(&ctx, "a"), FlatSymbolRefAttr::get(&ctx, "b"),
       FlatSymbolRefAttr::get(&ctx, "c"), FlatSymbolRefAttr::get(&ctx, "d")}
  ));
  ASSERT_EQ(debug::toStringOne(attr), "@a::@b::@c::@d");
}

TEST_F(SymbolHelperTests, test_asSymbolRefAttr_vector_Flat) {
  SymbolRefAttr attr = asSymbolRefAttr(
      std::vector(
          {FlatSymbolRefAttr::get(&ctx, "a"), FlatSymbolRefAttr::get(&ctx, "b"),
           FlatSymbolRefAttr::get(&ctx, "c"), FlatSymbolRefAttr::get(&ctx, "d")}
      )
  );
  ASSERT_EQ(debug::toStringOne(attr), "@a::@b::@c::@d");
}

TEST_F(SymbolHelperTests, test_getTailAsSymbolRefAttr) {
  SymbolRefAttr attr = getTailAsSymbolRefAttr(newExample(5));
  ASSERT_EQ(debug::toStringOne(attr), "@r1::@r2::@r3::@r4::@r5");
}

TEST_F(SymbolHelperTests, test_getPrefixAsSymbolRefAttr) {
  SymbolRefAttr attr = getPrefixAsSymbolRefAttr(newExample(5));
  ASSERT_EQ(debug::toStringOne(attr), "@root::@r1::@r2::@r3::@r4");
}

TEST_F(SymbolHelperTests, test_replaceLeaf) {
  SymbolRefAttr attr = replaceLeaf(newExample(2), "leaf");
  ASSERT_EQ(debug::toStringOne(attr), "@root::@r1::@leaf");
}

TEST_F(SymbolHelperTests, test_appendLeaf) {
  SymbolRefAttr attr = appendLeaf(newExample(2), "leaf");
  ASSERT_EQ(debug::toStringOne(attr), "@root::@r1::@r2::@leaf");
}

TEST_F(SymbolHelperTests, test_appendLeafName) {
  SymbolRefAttr attr = appendLeafName(newExample(2), "_suffix");
  ASSERT_EQ(debug::toStringOne(attr), "@root::@r1::@r2_suffix");
}

TEST_F(SymbolHelperTests, test_getPathRelativeToAncestor_nestedAncestors) {
  auto module = parseSourceString<ModuleOp>(
      R"mlir(
    module @Wrapper {
      module @Top attributes {llzk.lang} {
        module @Bar attributes {llzk.lang} {
          poly.template @TFoo {
            poly.param @N : index
            struct.def @Foo {
              function.def @compute() -> !struct.type<@TFoo::@Foo<[@N]>> {
                %s = struct.new : !struct.type<@TFoo::@Foo<[@N]>>
                function.return %s : !struct.type<@TFoo::@Foo<[@N]>>
              }
              function.def @constrain(%s: !struct.type<@TFoo::@Foo<[@N]>>) {
                function.return
              }
            }
          }
        }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);
  auto top = module->lookupSymbol<ModuleOp>("Top");
  auto bar = top.lookupSymbol<ModuleOp>("Bar");
  auto templ = bar.lookupSymbol<polymorphic::TemplateOp>("TFoo");
  auto foo = cast<component::StructDefOp>(SymbolTable::lookupSymbolIn(templ, "Foo"));

  auto fromTop = getPathRelativeToAncestor(cast<SymbolOpInterface>(foo.getOperation()), top);
  ASSERT_TRUE(succeeded(fromTop));
  EXPECT_EQ(debug::toStringOne(*fromTop), "@Bar::@TFoo::@Foo");
  EXPECT_EQ(SymbolTable::lookupSymbolIn(top, *fromTop), foo.getOperation());

  auto fromBar = getPathRelativeToAncestor(cast<SymbolOpInterface>(foo.getOperation()), bar);
  ASSERT_TRUE(succeeded(fromBar));
  EXPECT_EQ(debug::toStringOne(*fromBar), "@TFoo::@Foo");
  EXPECT_EQ(SymbolTable::lookupSymbolIn(bar, *fromBar), foo.getOperation());
}

TEST_F(SymbolHelperTests, test_getPathRelativeToAncestor_unnamedAncestor) {
  auto module = parseSourceString<ModuleOp>(
      R"mlir(
    module attributes {llzk.lang} {
      struct.def @Foo {
        function.def @compute() -> !struct.type<@Foo> {
          %s = struct.new : !struct.type<@Foo>
          function.return %s : !struct.type<@Foo>
        }
        function.def @constrain(%s: !struct.type<@Foo>) { function.return }
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);
  auto foo = module->lookupSymbol<component::StructDefOp>("Foo");
  auto path = getPathRelativeToAncestor(cast<SymbolOpInterface>(foo.getOperation()), *module);
  ASSERT_TRUE(succeeded(path));
  EXPECT_EQ(debug::toStringOne(*path), "@Foo");
  EXPECT_EQ(SymbolTable::lookupSymbolIn(*module, *path), foo.getOperation());
}

TEST_F(SymbolHelperTests, test_getPathRelativeToAncestor_inaccessibleSymbols) {
  auto module = parseSourceString<ModuleOp>(
      R"mlir(
    module @Top attributes {llzk.lang} {
      module @Left attributes {llzk.lang} {}
      module @Right attributes {llzk.lang} {
        module @Foo {}
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);
  auto left = module->lookupSymbol<ModuleOp>("Left");
  auto right = module->lookupSymbol<ModuleOp>("Right");
  auto foo = right.lookupSymbol<ModuleOp>("Foo");

  // Check the helper's location notes without automatic operation dumps.
  ctx.printOpOnDiagnostic(false);
  SmallVector<Diagnostic> diagnostics;
  ScopedDiagnosticHandler handler(&ctx, [&diagnostics](Diagnostic &diagnostic) {
    diagnostics.push_back(std::move(diagnostic));
    return success();
  });
  // Callers that do not request diagnostics retain the silent-failure behavior.
  EXPECT_TRUE(failed(getPathRelativeToAncestor(cast<SymbolOpInterface>(foo.getOperation()), left)));
  EXPECT_TRUE(diagnostics.empty());

  auto emitError = [left] { return left->emitError("requested symbol"); };
  EXPECT_TRUE(failed(
      getPathRelativeToAncestor(cast<SymbolOpInterface>(foo.getOperation()), left, emitError)
  ));
  EXPECT_TRUE(failed(
      getPathRelativeToAncestor(cast<SymbolOpInterface>(module->getOperation()), left, emitError)
  ));
  EXPECT_TRUE(failed(
      getPathRelativeToAncestor(cast<SymbolOpInterface>(left.getOperation()), left, emitError)
  ));
  ASSERT_EQ(diagnostics.size(), 3);
  for (const auto &diagnostic : diagnostics) {
    EXPECT_EQ(diagnostic.getLocation(), left.getLoc());
  }
  EXPECT_EQ(diagnostics[0].str(), "requested symbol is not visible from module \"Left\"");
  auto notes = diagnostics[0].getNotes();
  ASSERT_EQ(std::distance(notes.begin(), notes.end()), 2);
  EXPECT_EQ(notes.begin()->getLocation(), foo.getLoc());
  EXPECT_EQ(std::next(notes.begin())->getLocation(), left.getLoc());
  auto ancestorNotes = diagnostics[1].getNotes();
  ASSERT_EQ(std::distance(ancestorNotes.begin(), ancestorNotes.end()), 2);
  EXPECT_EQ(ancestorNotes.begin()->getLocation(), module->getLoc());
  EXPECT_EQ(std::next(ancestorNotes.begin())->getLocation(), left.getLoc());
  EXPECT_EQ(
      diagnostics[2].str(),
      "requested symbol cannot be named relative to itself; a strict descendant is required"
  );
  auto selfNotes = diagnostics[2].getNotes();
  ASSERT_EQ(std::distance(selfNotes.begin(), selfNotes.end()), 1);
  EXPECT_EQ(selfNotes.begin()->getLocation(), left.getLoc());
}

TEST_F(SymbolHelperTests, test_getPathRelativeToAncestor_unnamedIntermediateModule) {
  auto module = parseSourceString<ModuleOp>(
      R"mlir(
    module @Top attributes {llzk.lang} {
      module {
        module @Hidden {}
      }
    }
  )mlir",
      &ctx
  );
  ASSERT_TRUE(module);
  auto unnamed = cast<ModuleOp>(module->getBody()->front());
  auto hidden = unnamed.lookupSymbol<ModuleOp>("Hidden");
  // Check the helper's location notes without automatic operation dumps.
  ctx.printOpOnDiagnostic(false);
  SmallVector<Diagnostic> diagnostics;
  ScopedDiagnosticHandler handler(&ctx, [&diagnostics](Diagnostic &diagnostic) {
    diagnostics.push_back(std::move(diagnostic));
    return success();
  });
  auto emitError = [ancestor = *module] { return ancestor->emitError("requested symbol"); };
  EXPECT_TRUE(failed(
      getPathRelativeToAncestor(cast<SymbolOpInterface>(hidden.getOperation()), *module, emitError)
  ));
  ASSERT_EQ(diagnostics.size(), 1);
  EXPECT_EQ(diagnostics[0].getLocation(), module->getLoc());
  EXPECT_EQ(
      diagnostics[0].str(), "requested symbol cannot be named relative to module \"Top\" because "
                            "an intervening symbol table is unnamed"
  );
  auto notes = diagnostics[0].getNotes();
  ASSERT_EQ(std::distance(notes.begin(), notes.end()), 2);
  EXPECT_EQ(notes.begin()->getLocation(), unnamed.getLoc());
  EXPECT_EQ(std::next(notes.begin())->getLocation(), module->getLoc());
}
