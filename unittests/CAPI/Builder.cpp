//===-- Builder.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Builder.h"

#include "CAPITestBase.h"

#include <mlir-c/IR.h>

namespace {

MlirOperation createContainerOp(MlirContext context, MlirBlock *block) {
  MlirModule module = mlirModuleCreateEmpty(mlirLocationUnknownGet(context));
  *block = mlirModuleGetBody(module);
  return mlirModuleGetOperation(module);
}

MlirOperation getNthOperation(MlirBlock block, unsigned index) {
  MlirOperation op = mlirBlockGetFirstOperation(block);
  for (unsigned i = 0; i < index; ++i) {
    op = mlirOperationGetNextInBlock(op);
  }
  return op;
}

} // namespace

TEST_F(CAPITest, MlirOpBuilderCreate) {
  auto builder = mlirOpBuilderCreate(context);
  mlirOpBuilderDestroy(builder);
}

static void test_cb1(MlirOperation, MlirOpBuilderInsertPoint, void *) {}
static void test_cb2(MlirBlock, MlirRegion, MlirBlock, void *) {}

TEST_F(CAPITest, MlirOpBuilderCreateWithListener) {
  auto listener = mlirOpBuilderListenerCreate(test_cb1, test_cb2, NULL);
  auto builder = mlirOpBuilderCreateWithListener(context, listener);
  mlirOpBuilderDestroy(builder);
  mlirOpBuilderListenerDestroy(listener);
}

TEST_F(CAPITest, MlirOpBuilderListenerCreate) {
  auto listener = mlirOpBuilderListenerCreate(test_cb1, test_cb2, NULL);
  mlirOpBuilderListenerDestroy(listener);
}

TEST_F(CAPITest, MlirOpBuilderGetContext) {
  auto builder = mlirOpBuilderCreate(context);
  EXPECT_TRUE(mlirContextEqual(mlirOpBuilderGetContext(builder), context));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, MlirOpBuilderSetInsertionPointToStart) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointToStart(builder, block);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), first));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderSetInsertionPointToEnd) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointToEnd(builder, block);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_EQ(mlirOpBuilderGetInsertionPoint(builder).ptr, nullptr);

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderSetInsertionPoint) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPoint(builder, second);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderSetInsertionPointAfter) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointAfter(builder, first);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderSetInsertionPointAfterValue) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointAfterValue(builder, mlirOperationGetResult(first, 0));
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderSaveInsertionPoint) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPoint(builder, second);
  MlirOpBuilderInsertPoint saved = mlirOpBuilderSaveInsertionPoint(builder);
  EXPECT_EQ(saved.block.ptr, block.ptr);
  EXPECT_TRUE(mlirOperationEqual(saved.point, second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderSaveInsertionPointWithoutOperation) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointToEnd(builder, block);
  MlirOpBuilderInsertPoint saved = mlirOpBuilderSaveInsertionPoint(builder);
  EXPECT_EQ(saved.block.ptr, block.ptr);
  EXPECT_EQ(saved.point.ptr, nullptr);

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderRestoreInsertionPoint) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPoint(builder, second);
  MlirOpBuilderInsertPoint saved = mlirOpBuilderSaveInsertionPoint(builder);
  mlirOpBuilderSetInsertionPointToStart(builder, block);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), first));
  mlirOpBuilderRestoreInsertionPoint(builder, saved);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderRestoreInsertionPointWithoutOperation) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  MlirOperation third = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointToEnd(builder, block);
  MlirOpBuilderInsertPoint saved = mlirOpBuilderSaveInsertionPoint(builder);
  EXPECT_EQ(saved.block.ptr, block.ptr);
  EXPECT_EQ(saved.point.ptr, nullptr);
  mlirOpBuilderSetInsertionPointToStart(builder, block);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderGetInsertionPoint(builder), first));
  mlirOpBuilderRestoreInsertionPoint(builder, saved);
  mlirOpBuilderInsert(builder, third);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, block.ptr);
  EXPECT_EQ(mlirOpBuilderGetInsertionPoint(builder).ptr, nullptr);
  EXPECT_TRUE(mlirOperationEqual(mlirOperationGetNextInBlock(second), third));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderClearInsertionPoint) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);

  auto builder = mlirOpBuilderCreate(context);

  mlirOpBuilderSetInsertionPointToStart(builder, block);
  mlirOpBuilderClearInsertionPoint(builder);
  EXPECT_EQ(mlirOpBuilderGetInsertionBlock(builder).ptr, nullptr);
  EXPECT_EQ(mlirOpBuilderGetInsertionPoint(builder).ptr, nullptr);

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderInsertAtStart) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  MlirOperation insertAtStart = createIndexOperation();
  mlirOpBuilderSetInsertionPointToStart(builder, block);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderInsert(builder, insertAtStart), insertAtStart));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 0), insertAtStart));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 1), first));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderInsertBeforeOperation) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  MlirOperation insertBeforeSecond = createIndexOperation();
  mlirOpBuilderSetInsertionPoint(builder, second);
  EXPECT_TRUE(
      mlirOperationEqual(mlirOpBuilderInsert(builder, insertBeforeSecond), insertBeforeSecond)
  );
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 1), insertBeforeSecond));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 2), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderInsertAfterOperation) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  MlirOperation insertAfterFirst = createIndexOperation();
  mlirOpBuilderSetInsertionPointAfter(builder, first);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderInsert(builder, insertAfterFirst), insertAfterFirst));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 1), insertAfterFirst));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 2), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderInsertAfterValue) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  MlirOperation insertAfterValue = createIndexOperation();
  mlirOpBuilderSetInsertionPointAfterValue(builder, mlirOperationGetResult(first, 0));
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderInsert(builder, insertAfterValue), insertAfterValue));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 1), insertAfterValue));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 2), second));

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}

TEST_F(CAPITest, MlirOpBuilderInsertAtEnd) {
  MlirBlock block;
  MlirOperation module = createContainerOp(context, &block);
  MlirOperation first = createIndexOperation();
  MlirOperation second = createIndexOperation();
  mlirBlockAppendOwnedOperation(block, first);
  mlirBlockAppendOwnedOperation(block, second);

  auto builder = mlirOpBuilderCreate(context);

  MlirOperation insertAtEnd = createIndexOperation();
  mlirOpBuilderSetInsertionPointToEnd(builder, block);
  EXPECT_TRUE(mlirOperationEqual(mlirOpBuilderInsert(builder, insertAtEnd), insertAtEnd));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 1), second));
  EXPECT_TRUE(mlirOperationEqual(getNthOperation(block, 2), insertAtEnd));
  EXPECT_EQ(mlirOperationGetNextInBlock(insertAtEnd).ptr, nullptr);

  mlirOpBuilderDestroy(builder);
  mlirOperationDestroy(module);
}
