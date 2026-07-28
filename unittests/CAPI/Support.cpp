//===-- Support.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Support.h"

#include "CAPITestBase.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include <llvm/ADT/ArrayRef.h>

#include <cstdint>
#include <gtest/gtest.h>

TEST_F(CAPITest, MlirOperationReplaceUsesOfWith) {
  // Create two constant operations that produce values
  MlirOperation const1 = createIndexOperation();
  MlirOperation const2 = createIndexOperation();
  MlirValue value1 = mlirOperationGetResult(const1, 0);
  MlirValue value2 = mlirOperationGetResult(const2, 0);

  // Create an operation that uses value1: `arith.addi value1, value1 : index`
  MlirType indexType = createIndexType();
  MlirOperationState addState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("arith.addi"), mlirLocationUnknownGet(context)
  );
  mlirOperationStateAddResults(&addState, 1, &indexType);
  MlirValue operands[2] = {value1, value1};
  mlirOperationStateAddOperands(&addState, 2, operands);
  MlirOperation addOp = mlirOperationCreate(&addState);

  // Verify that the add operation uses value1 twice
  EXPECT_EQ(mlirOperationGetNumOperands(addOp), 2);
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 0), value1));
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 1), value1));

  // Replace uses of value1 with value2 inside the add operation
  mlirOperationReplaceUsesOfWith(addOp, value1, value2);

  // Verify that both operands now use value2
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 0), value2));
  EXPECT_TRUE(mlirValueEqual(mlirOperationGetOperand(addOp, 1), value2));

  // Clean up
  mlirOperationDestroy(addOp);
  mlirOperationDestroy(const2);
  mlirOperationDestroy(const1);
}

TEST_F(CAPITest, MlirOperationWalkReverse) {
  // Track visited operations
  std::vector<MlirOperation> visitedOps;

  // Define the callback function
  auto callback = [](MlirOperation op, void *userData) -> MlirWalkResult {
    auto *ops = static_cast<std::vector<MlirOperation> *>(userData);
    ops->push_back(op);
    return MlirWalkResultAdvance;
  };

  // Create a nested structure: module { func { block with 3 operations } }
  MlirOperation const1 = createIndexOperation();
  MlirOperation const2 = createIndexOperation();
  MlirOperation const3 = createIndexOperation();

  // Create a block and add operations to it
  MlirRegion region = mlirRegionCreate();
  MlirBlock block = mlirBlockCreate(0, nullptr, nullptr);
  mlirRegionAppendOwnedBlock(region, block);
  mlirBlockAppendOwnedOperation(block, const1);
  mlirBlockAppendOwnedOperation(block, const2);
  mlirBlockAppendOwnedOperation(block, const3);

  // Create a module operation to hold the region
  MlirOperationState parentState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("builtin.module"), mlirLocationUnknownGet(context)
  );
  mlirOperationStateAddOwnedRegions(&parentState, 1, &region);
  MlirOperation parentOp = mlirOperationCreate(&parentState);

  // Walk in reverse order (PostOrder)
  mlirOperationWalkReverse(parentOp, callback, &visitedOps, MlirWalkPostOrder);

  // In reverse PostOrder, children are visited before parent, but in reverse order
  // Expected order: const3, const2, const1, parentOp
  EXPECT_EQ(visitedOps.size(), 4);
  EXPECT_TRUE(mlirOperationEqual(visitedOps[0], const3));
  EXPECT_TRUE(mlirOperationEqual(visitedOps[1], const2));
  EXPECT_TRUE(mlirOperationEqual(visitedOps[2], const1));
  EXPECT_TRUE(mlirOperationEqual(visitedOps[3], parentOp));

  // Test reverse PreOrder walk
  visitedOps.clear();
  mlirOperationWalkReverse(parentOp, callback, &visitedOps, MlirWalkPreOrder);

  // In reverse PreOrder, parent is visited before children, and children in reverse
  // Expected order: parentOp, const3, const2, const1
  EXPECT_EQ(visitedOps.size(), 4);
  EXPECT_TRUE(mlirOperationEqual(visitedOps[0], parentOp));
  EXPECT_TRUE(mlirOperationEqual(visitedOps[1], const3));
  EXPECT_TRUE(mlirOperationEqual(visitedOps[2], const2));
  EXPECT_TRUE(mlirOperationEqual(visitedOps[3], const1));
  // Clean up
  mlirOperationDestroy(parentOp);
}

TEST_F(CAPITest, LlzkSymbolTableInsert) {
  MlirOperationState moduleState = mlirOperationStateGet(
      mlirStringRefCreateFromCString("builtin.module"), mlirLocationUnknownGet(context)
  );
  MlirRegion moduleRegion = mlirRegionCreate();
  MlirBlock moduleBlock = mlirBlockCreate(0, nullptr, nullptr);
  mlirRegionAppendOwnedBlock(moduleRegion, moduleBlock);
  mlirOperationStateAddOwnedRegions(&moduleState, 1, &moduleRegion);
  MlirOperation moduleOp = mlirOperationCreate(&moduleState);

  MlirOperationState symbolState1 = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.func"), mlirLocationUnknownGet(context)
  );
  MlirNamedAttribute symName1 = mlirNamedAttributeGet(
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("sym_name")),
      mlirStringAttrGet(context, mlirStringRefCreateFromCString("foo"))
  );
  MlirType funcType = mlirFunctionTypeGet(context, 0, nullptr, 0, nullptr);
  MlirNamedAttribute funcTypeAttr1 = mlirNamedAttributeGet(
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("function_type")),
      mlirTypeAttrGet(funcType)
  );
  MlirNamedAttribute symbolAttrs1[2] = {symName1, funcTypeAttr1};
  mlirOperationStateAddAttributes(&symbolState1, 2, symbolAttrs1);
  MlirRegion symbolRegion1 = mlirRegionCreate();
  MlirBlock symbolBlock1 = mlirBlockCreate(0, nullptr, nullptr);
  mlirRegionAppendOwnedBlock(symbolRegion1, symbolBlock1);
  mlirOperationStateAddOwnedRegions(&symbolState1, 1, &symbolRegion1);
  MlirOperation symbol1 = mlirOperationCreate(&symbolState1);
  mlirBlockAppendOwnedOperation(moduleBlock, symbol1);

  MlirOperationState symbolState2 = mlirOperationStateGet(
      mlirStringRefCreateFromCString("func.func"), mlirLocationUnknownGet(context)
  );
  MlirNamedAttribute symName2 = mlirNamedAttributeGet(
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("sym_name")),
      mlirStringAttrGet(context, mlirStringRefCreateFromCString("foo"))
  );
  MlirNamedAttribute funcTypeAttr2 = mlirNamedAttributeGet(
      mlirIdentifierGet(context, mlirStringRefCreateFromCString("function_type")),
      mlirTypeAttrGet(funcType)
  );
  MlirNamedAttribute symbolAttrs2[2] = {symName2, funcTypeAttr2};
  mlirOperationStateAddAttributes(&symbolState2, 2, symbolAttrs2);
  MlirRegion symbolRegion2 = mlirRegionCreate();
  MlirBlock symbolBlock2 = mlirBlockCreate(0, nullptr, nullptr);
  mlirRegionAppendOwnedBlock(symbolRegion2, symbolBlock2);
  mlirOperationStateAddOwnedRegions(&symbolState2, 1, &symbolRegion2);
  MlirOperation symbol2 = mlirOperationCreate(&symbolState2);

  llzkSymbolTableInsert(moduleOp, symbol2);

  EXPECT_TRUE(mlirOperationEqual(mlirOperationGetParentOperation(symbol2), moduleOp));
  MlirAttribute insertedNameAttr =
      mlirOperationGetAttributeByName(symbol2, mlirStringRefCreateFromCString("sym_name"));
  EXPECT_TRUE(mlirAttributeIsAString(insertedNameAttr));
  EXPECT_FALSE(mlirStringRefEqual(
      mlirStringAttrGetValue(insertedNameAttr), mlirStringRefCreateFromCString("foo")
  ));

  mlirOperationDestroy(moduleOp);
}

class LlzkAffineMapOperandsBuilderTests : public ::testing::Test {
protected:
  MlirContext context;
  LlzkAffineMapOperandsBuilder builder;
  MlirValueRange maps[3];
  int32_t dims[3] = {1, 2, 0};

  LlzkAffineMapOperandsBuilderTests()
      : context(mlirContextCreate()), builder(llzkAffineMapOperandsBuilderCreate()) {
    int32_t sizes[] = {2, 3, 1};
    for (auto i = 0; i < 3; i++) {
      auto size = sizes[i];
      maps[i] = MlirValueRange {.values = new MlirValue[size], .size = size};
    }
  }

  ~LlzkAffineMapOperandsBuilderTests() override {
    llzkAffineMapOperandsBuilderDestroy(&builder);
    mlirContextDestroy(context);
    for (auto i = 0; i < 3; i++) {
      delete[] maps[i].values;
    }
  }
};

TEST_F(LlzkAffineMapOperandsBuilderTests, AppendOperandsOnce) {
  llzkAffineMapOperandsBuilderAppendOperands(&builder, 2, maps);
  ASSERT_EQ(builder.nMapOperands, 2);
}

TEST_F(LlzkAffineMapOperandsBuilderTests, AppendOperandsTwice) {
  llzkAffineMapOperandsBuilderAppendOperands(&builder, 2, maps);
  ASSERT_EQ(builder.nMapOperands, 2);
  llzkAffineMapOperandsBuilderAppendOperands(&builder, 1, maps + 2);
  ASSERT_EQ(builder.nMapOperands, 3);
}

TEST_F(LlzkAffineMapOperandsBuilderTests, AppendOperandsAndDimsOnce) {
  llzkAffineMapOperandsBuilderAppendOperandsWithDimCount(&builder, 2, maps, dims);
  ASSERT_EQ(builder.nMapOperands, 2);
  ASSERT_EQ(builder.nDimsPerMap, 2);
}

TEST_F(LlzkAffineMapOperandsBuilderTests, AppendOperandsAndDimsTwice) {
  llzkAffineMapOperandsBuilderAppendOperandsWithDimCount(&builder, 2, maps, dims);
  ASSERT_EQ(builder.nMapOperands, 2);
  ASSERT_EQ(builder.nDimsPerMap, 2);
  llzkAffineMapOperandsBuilderAppendOperandsWithDimCount(&builder, 1, maps + 2, dims + 2);
  ASSERT_EQ(builder.nMapOperands, 3);
  ASSERT_EQ(builder.nDimsPerMap, 3);
}

TEST_F(LlzkAffineMapOperandsBuilderTests, AppendOperandsAndDimsSeparateOnce) {
  llzkAffineMapOperandsBuilderAppendOperands(&builder, 2, maps);
  ASSERT_EQ(builder.nMapOperands, 2);
  ASSERT_EQ(builder.nDimsPerMap, 0);
  llzkAffineMapOperandsBuilderAppendDimCount(&builder, 2, dims);
  ASSERT_EQ(builder.nDimsPerMap, 2);
}

TEST_F(LlzkAffineMapOperandsBuilderTests, AppendOperandsAndDimsSeparateTwice) {
  llzkAffineMapOperandsBuilderAppendOperands(&builder, 2, maps);
  ASSERT_EQ(builder.nMapOperands, 2);
  ASSERT_EQ(builder.nDimsPerMap, 0);
  llzkAffineMapOperandsBuilderAppendDimCount(&builder, 2, dims);
  ASSERT_EQ(builder.nDimsPerMap, 2);
  llzkAffineMapOperandsBuilderAppendOperands(&builder, 1, maps + 2);
  ASSERT_EQ(builder.nMapOperands, 3);
  llzkAffineMapOperandsBuilderAppendDimCount(&builder, 1, dims + 2);
  ASSERT_EQ(builder.nDimsPerMap, 3);
}

TEST_F(LlzkAffineMapOperandsBuilderTests, ConversionToAttr) {
  ASSERT_EQ(builder.nDimsPerMap, 0);
  llzkAffineMapOperandsBuilderAppendDimCount(&builder, 3, dims);
  ASSERT_EQ(builder.nDimsPerMap, 3);
  llzkAffineMapOperandsBuilderConvertDimsPerMapToAttr(&builder, context);
  ASSERT_EQ(builder.nDimsPerMap, -1);
  ASSERT_TRUE(mlirAttributeIsADenseI32Array(builder.dimsPerMap.attr));
}

TEST_F(LlzkAffineMapOperandsBuilderTests, ConversionToAttrAndBackToArray) {
  ASSERT_EQ(builder.nDimsPerMap, 0);
  llzkAffineMapOperandsBuilderAppendDimCount(&builder, 3, dims);
  ASSERT_EQ(builder.nDimsPerMap, 3);
  llzkAffineMapOperandsBuilderConvertDimsPerMapToAttr(&builder, context);
  ASSERT_EQ(builder.nDimsPerMap, -1);
  ASSERT_TRUE(mlirAttributeIsADenseI32Array(builder.dimsPerMap.attr));
  llzkAffineMapOperandsBuilderConvertDimsPerMapToArray(&builder);
  ASSERT_EQ(builder.nDimsPerMap, 3);
  ASSERT_EQ(llvm::ArrayRef(builder.dimsPerMap.array, builder.nDimsPerMap), llvm::ArrayRef(dims, 3));
}
