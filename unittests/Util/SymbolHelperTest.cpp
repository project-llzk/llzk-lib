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

#include "llzk/Dialect/Shared/Builders.h"
#include "llzk/Util/Debug.h"

#include <mlir/IR/BuiltinAttributes.h>

#include <gtest/gtest.h>

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
