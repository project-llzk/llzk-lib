//===-- GlobalDialectTests.cpp - Global dialect bytecode tests --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Global/IR/Dialect.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/MemoryBufferRef.h>

#include <gtest/gtest.h>

using namespace llzk;
using namespace mlir;

namespace {

class GlobalDialectBytecodeTests : public LLZKTest {};

std::string writeLegacyBooleanArrayBytecode(MLIRContext &ctx, uint64_t value) {
  OwningOpRef<ModuleOp> module = createLLZKModule(&ctx);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(module->getBody());

  Type boolType = builder.getI1Type();
  auto arrayType = llzk::array::ArrayType::get(boolType, {2});
  Attribute legacyValue = IntegerAttr::get(builder.getI64Type(), value);
  auto initializer = ArrayAttr::get(&ctx, {legacyValue, legacyValue});
  builder.create<llzk::global::GlobalDefOp>(
      UnknownLoc::get(&ctx), "bits", /*constant=*/true, arrayType, initializer
  );

  BytecodeWriterConfig config;
  config.setDialectVersion<llzk::global::GlobalDialect>(
      std::make_unique<LLZKDialectVersion>(2, 1, 2)
  );
  std::string bytecode;
  llvm::raw_string_ostream stream(bytecode);
  EXPECT_TRUE(succeeded(writeBytecodeToFile(module.get(), stream, config)));
  return bytecode;
}

std::string writeLegacyIndexArrayBytecode(MLIRContext &ctx, uint64_t value) {
  OwningOpRef<ModuleOp> module = createLLZKModule(&ctx);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToStart(module->getBody());

  Type indexType = builder.getIndexType();
  auto arrayType = llzk::array::ArrayType::get(indexType, {2});
  Attribute legacyValue = IntegerAttr::get(builder.getI8Type(), value);
  auto initializer = ArrayAttr::get(&ctx, {legacyValue, legacyValue});
  builder.create<llzk::global::GlobalDefOp>(
      UnknownLoc::get(&ctx), "indices", /*constant=*/true, arrayType, initializer
  );

  BytecodeWriterConfig config;
  config.setDialectVersion<llzk::global::GlobalDialect>(
      std::make_unique<LLZKDialectVersion>(2, 1, 2)
  );
  std::string bytecode;
  llvm::raw_string_ostream stream(bytecode);
  EXPECT_TRUE(succeeded(writeBytecodeToFile(module.get(), stream, config)));
  return bytecode;
}

TEST_F(GlobalDialectBytecodeTests, UpgradesBooleanArrayPayloadWidth) {
  std::string bytecode = writeLegacyBooleanArrayBytecode(ctx, 1);
  Block block;
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "legacy-boolean-array"), &block, ParserConfig(&ctx)
  )));

  auto module = llvm::cast<ModuleOp>(block.front());
  auto global = *module.getOps<llzk::global::GlobalDefOp>().begin();
  auto initializer = llvm::cast<ArrayAttr>(global.getInitialValueAttr());
  for (Attribute element : initializer) {
    auto integer = llvm::cast<IntegerAttr>(element);
    EXPECT_EQ(integer.getType(), IntegerType::get(&ctx, 1));
    EXPECT_EQ(integer.getValue().getBitWidth(), 1U);
    EXPECT_TRUE(integer.getValue().isOne());
  }
}

TEST_F(GlobalDialectBytecodeTests, RejectsOutOfRangeLegacyBooleanArrayValue) {
  std::string bytecode = writeLegacyBooleanArrayBytecode(ctx, 2);
  Block block;
  EXPECT_TRUE(failed(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "invalid-legacy-boolean-array"), &block, ParserConfig(&ctx)
  )));
}

TEST_F(GlobalDialectBytecodeTests, UpgradesIndexArrayPayloadWidth) {
  std::string bytecode = writeLegacyIndexArrayBytecode(ctx, 1);
  Block block;
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "legacy-index-array"), &block, ParserConfig(&ctx)
  )));

  auto module = llvm::cast<ModuleOp>(block.front());
  auto global = *module.getOps<llzk::global::GlobalDefOp>().begin();
  auto initializer = llvm::cast<ArrayAttr>(global.getInitialValueAttr());
  for (Attribute element : initializer) {
    auto integer = llvm::cast<IntegerAttr>(element);
    EXPECT_EQ(integer.getType(), IndexType::get(&ctx));
    EXPECT_EQ(integer.getValue().getBitWidth(), IndexType::kInternalStorageBitWidth);
    EXPECT_TRUE(integer.getValue().isOne());
  }

  std::string reserialized;
  llvm::raw_string_ostream stream(reserialized);
  EXPECT_TRUE(succeeded(writeBytecodeToFile(module, stream)));
}

} // namespace
