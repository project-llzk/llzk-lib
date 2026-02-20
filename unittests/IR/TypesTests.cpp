//===-- TypesTests.cpp - Unit tests for LLZK types --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"

#include <gtest/gtest.h>

#include "../LLZKTestBase.h"

using namespace llzk;
using namespace mlir;
using namespace llzk::array;
using namespace llzk::component;
using namespace llzk::felt;
using namespace llzk::polymorphic;

class TypeTests : public LLZKTest {
protected:
  TypeTests() : LLZKTest() {}
};

TEST_F(TypeTests, testArrayTypeCloneSuccessNewType) {
  IntegerType tyBool = IntegerType::get(&ctx, 1);
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  ArrayType b = a.cloneWith(std::nullopt, tyBool);
  ASSERT_EQ(b.getElementType(), tyBool);
  ASSERT_EQ(b.getShape(), ArrayRef(std::vector<int64_t>({2, 2})));
}

TEST_F(TypeTests, testArrayTypeCloneSuccessNewShape) {
  IndexType tyIndex = IndexType::get(&ctx);
  ArrayType a = ArrayType::get(tyIndex, {2, 2});
  std::vector<int64_t> newShapeVec({2, 3, 2});
  ArrayRef newShape(newShapeVec);
  ArrayType b = a.cloneWith(std::make_optional(newShape), tyIndex);
  ASSERT_EQ(b.getElementType(), tyIndex);
  ASSERT_EQ(b.getShape(), newShape);
}

// The verification that throws the error is only active in debug builds.
// See:
// https://github.com/llvm/llvm-project/blob/87f0227cb60147a26a1eeb4fb06e3b505e9c7261/mlir/include/mlir/IR/StorageUniquerSupport.h#L179
#ifndef NDEBUG
TEST_F(TypeTests, testArrayTypeCloneWithEmptyShapeError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        ArrayType a = ArrayType::get(tyIndex, {2, 2});
        std::vector<int64_t> newShapeVec;
        ArrayRef newShape(newShapeVec);
        a.cloneWith(std::make_optional(newShape), tyIndex);
      },
      "error: array must have at least one dimension"
  );
}
#endif

TEST_F(TypeTests, testArrayTypeGetWithAttributeEmptyShapeError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        std::vector<Attribute> newDimsVec;
        ArrayRef<Attribute> dimensionSizes(newDimsVec);
        if (ArrayType() == ArrayType::get(tyIndex, dimensionSizes)) {
          // Force the error to be reported even when compiled in release mode
          std::abort();
        }
      },
      "error: array must have at least one dimension"
  );
}

TEST_F(TypeTests, testArrayTypeGetWithAttributeWrongAttrKindError) {
  EXPECT_DEATH(
      {
        IndexType tyIndex = IndexType::get(&ctx);
        std::vector<Attribute> newDimsVec = {UnitAttr::get(&ctx)};
        ArrayRef<Attribute> dimensionSizes(newDimsVec);
        if (ArrayType() == ArrayType::get(tyIndex, dimensionSizes)) {
          // Force the error to be reported even when compiled in release mode
          std::abort();
        }
      },
      "error: Array dimension must be one of .* but found 'builtin.unit'"
  );
}

TEST_F(TypeTests, testStructTypeIsConcreteNoParams) {
  StructType t = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"));
  // When the StructType has no parameters, isConcreteType() passes regardless of
  // `allowStructParams` flag.
  ASSERT_TRUE(isConcreteType(t, true));
  ASSERT_TRUE(isConcreteType(t, false));
}

TEST_F(TypeTests, testStructTypeIsConcreteWithParams) {
  Attribute a1 = IntegerAttr::get(IndexType::get(&ctx), 128);
  Attribute a2 = IntegerAttr::get(IndexType::get(&ctx), 15);
  StructType t = StructType::get(FlatSymbolRefAttr::get(&ctx, "TheName"), ArrayRef {a1, a2});
  // When the StructType has parameters, isConcreteType() passes when `allowStructParams` flag is
  // `true` but fails when it is `false`.
  ASSERT_TRUE(isConcreteType(t, true));
  ASSERT_FALSE(isConcreteType(t, false));
}

TEST_F(TypeTests, testShortString) {
  OpBuilder bldr(&ctx);
  EXPECT_EQ("b", BuildShortTypeString::from(bldr.getIntegerType(1)));
  EXPECT_EQ("i", BuildShortTypeString::from(bldr.getIndexType()));
  EXPECT_EQ(
      "!t<@A>", BuildShortTypeString::from(TypeVarType::get(FlatSymbolRefAttr::get(&ctx, "A")))
  );
  EXPECT_EQ(
      "!a<b:4_235_123>", BuildShortTypeString::from(
                             ArrayType::get(bldr.getIntegerType(1), ArrayRef<int64_t> {4, 235, 123})
                         )
  );
  EXPECT_EQ(
      "!s<@S1>", BuildShortTypeString::from(StructType::get(FlatSymbolRefAttr::get(&ctx, "S1")))
  );
  EXPECT_EQ(
      "!s<@S1_43>",
      BuildShortTypeString::from(
          StructType::get(
              FlatSymbolRefAttr::get(&ctx, "S1"),
              ArrayAttr::get(
                  &ctx, ArrayRef<Attribute> {bldr.getIntegerAttr(bldr.getIndexType(), 43)}
              )
          )
      )
  );
  {
    auto innerStruct = StructType::get(
        FlatSymbolRefAttr::get(&ctx, "S1"),
        ArrayAttr::get(&ctx, ArrayRef<Attribute> {bldr.getIntegerAttr(bldr.getIndexType(), 43)})
    );
    auto params = ArrayAttr::get(
        &ctx,
        ArrayRef<Attribute> {
            bldr.getIntegerAttr(bldr.getIndexType(), 43), FlatSymbolRefAttr::get(&ctx, "ParamName"),
            TypeAttr::get(ArrayType::get(FeltType::get(&ctx), ArrayRef<int64_t> {3, 5, 1, 5, 7})),
            TypeAttr::get(innerStruct), AffineMapAttr::get(bldr.getDimIdentityMap())
        }
    );
    EXPECT_EQ(
        "!s<@Top_43_@ParamName_!a<f:3_5_1_5_7>_!s<@S1_43>_!m<(d0)->(d0)>>",
        BuildShortTypeString::from(StructType::get(FlatSymbolRefAttr::get(&ctx, "Top"), params))
    );
  }

  // No protection/escaping of special characters in the original name
  EXPECT_EQ(
      "!s<@S1_!a<>>",
      BuildShortTypeString::from(StructType::get(FlatSymbolRefAttr::get(&ctx, "S1_!a<>")))
  );

  // Empty string produces "?"
  EXPECT_EQ("?", BuildShortTypeString::from(FlatSymbolRefAttr::get(&ctx, "")));
  EXPECT_EQ("?", BuildShortTypeString::from(FlatSymbolRefAttr::get(&ctx, StringRef())));

  {
    constexpr char withNull[] = {'a', 'b', '\0', 'c', 'd'};
    EXPECT_EQ(
        "5_@head_@ab_@Good_2",
        // clang-format off
        BuildShortTypeString::from(ArrayAttr::get(&ctx, ArrayRef<Attribute> {
          bldr.getIntegerAttr(bldr.getIndexType(), 5),
          FlatSymbolRefAttr::get(&ctx, "head\0_tail"),
          FlatSymbolRefAttr::get(&ctx, withNull),
          FlatSymbolRefAttr::get(&ctx, "Good"),
          bldr.getIntegerAttr(bldr.getIndexType(), 2)
        }))
        // clang-format on
    );
  }
}

TEST_F(TypeTests, testShortStringWithPartials) {
  auto symA = FlatSymbolRefAttr::get(&ctx, "A");
  auto symB = FlatSymbolRefAttr::get(&ctx, "B");
  auto symC = FlatSymbolRefAttr::get(&ctx, "C");
  auto symD = FlatSymbolRefAttr::get(&ctx, "D");
  auto symE = FlatSymbolRefAttr::get(&ctx, "E");
  auto symF = FlatSymbolRefAttr::get(&ctx, "F");
  auto symG = FlatSymbolRefAttr::get(&ctx, "G");
  auto symH = FlatSymbolRefAttr::get(&ctx, "H");
  auto symJ = FlatSymbolRefAttr::get(&ctx, "J");
  auto symK = FlatSymbolRefAttr::get(&ctx, "K");

  std::string v1 = BuildShortTypeString::from(
      "prefix", ArrayRef<Attribute> {
                    nullptr, symA, nullptr, nullptr, symB, nullptr, nullptr, nullptr, symC, nullptr
                }
  );
  EXPECT_EQ("prefix_\x1A_@A_\x1A_\x1A_@B_\x1A_\x1A_\x1A_@C_\x1A", v1);

  std::string v2 = BuildShortTypeString::from(
      v1, ArrayRef<Attribute> {nullptr, nullptr, symD, nullptr, symE, symF, nullptr}
  );
  EXPECT_EQ("prefix_\x1A_@A_\x1A_@D_@B_\x1A_@E_@F_@C_\x1A", v2);

  std::string v3 =
      BuildShortTypeString::from(v2, ArrayRef<Attribute> {symG, nullptr, nullptr, symH});
  EXPECT_EQ("prefix_@G_@A_\x1A_@D_@B_\x1A_@E_@F_@C_@H", v3);

  std::string v4 = BuildShortTypeString::from(v3, ArrayRef<Attribute> {symJ, symK});
  EXPECT_EQ("prefix_@G_@A_@J_@D_@B_@K_@E_@F_@C_@H", v4);
}

TEST_F(TypeTests, testShortStringWithPartials_withExtensions) {
  auto symA = FlatSymbolRefAttr::get(&ctx, "A");
  auto symB = FlatSymbolRefAttr::get(&ctx, "B");
  auto symC = FlatSymbolRefAttr::get(&ctx, "C");
  auto symD = FlatSymbolRefAttr::get(&ctx, "D");
  auto symE = FlatSymbolRefAttr::get(&ctx, "E");
  auto symF = FlatSymbolRefAttr::get(&ctx, "F");
  auto symG = FlatSymbolRefAttr::get(&ctx, "G");
  auto symH = FlatSymbolRefAttr::get(&ctx, "H");
  auto symJ = FlatSymbolRefAttr::get(&ctx, "J");
  auto symK = FlatSymbolRefAttr::get(&ctx, "K");

  std::string v1 = BuildShortTypeString::from(
      "prefix", ArrayRef<Attribute> {nullptr, symA, nullptr, nullptr, symB}
  );
  EXPECT_EQ("prefix_\x1A_@A_\x1A_\x1A_@B", v1);

  std::string v2 =
      BuildShortTypeString::from(v1, ArrayRef<Attribute> {nullptr, nullptr, symC, nullptr, symD});
  EXPECT_EQ("prefix_\x1A_@A_\x1A_@C_@B_\x1A_@D", v2);

  std::string v3 =
      BuildShortTypeString::from(v2, ArrayRef<Attribute> {symE, nullptr, nullptr, symF});
  EXPECT_EQ("prefix_@E_@A_\x1A_@C_@B_\x1A_@D_@F", v3);

  std::string v4 = BuildShortTypeString::from(v3, ArrayRef<Attribute> {symG, symH, symJ, symK});
  EXPECT_EQ("prefix_@E_@A_@G_@C_@B_@H_@D_@F_@J_@K", v4);
}
