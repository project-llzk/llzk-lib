//===-- GlobalDialectTests.cpp - Unit tests for global dialect --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"

#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::global;

namespace {

class GlobalDialectTests : public LLZKTest {};

TEST_F(GlobalDialectTests, BytecodeUpgradeNormalizesFeltInitializers) {
  constexpr StringLiteral source = R"mlir(
module attributes {llzk.lang} {
  "global.def"() <{constant, initial_value = #felt<const 1 : !felt.type<"bn128">>, sym_name = "refined", type = !felt.type}> : () -> ()
  "global.def"() <{constant, initial_value = #felt<const 2>, sym_name = "adopted", type = !felt.type<"bn128">}> : () -> ()
  "global.def"() <{constant, initial_value = [#felt<const 3 : !felt.type<"bn128">>, #felt<const 4>], sym_name = "array_refined", type = !array.type<2 x !felt.type>}> : () -> ()
  "global.def"() <{constant, initial_value = [#felt<const 5>, #felt<const 6>], sym_name = "array_adopted", type = !array.type<2 x !felt.type<"bn128">>}> : () -> ()
}
)mlir";

  auto legacy = parseSourceString<ModuleOp>(source, ParserConfig(&ctx, /*verifyAfterParse=*/false));
  ASSERT_TRUE(legacy);

  std::string bytecode;
  llvm::raw_string_ostream stream(bytecode);
  BytecodeWriterConfig writerConfig;
  writerConfig.setDialectVersion<GlobalDialect>(std::make_unique<LLZKDialectVersion>(0, 0, 0));
  ASSERT_TRUE(succeeded(writeBytecodeToFile(legacy.get(), stream, writerConfig)));
  stream.flush();

  Block upgradedBlock;
  ASSERT_TRUE(succeeded(readBytecodeFile(
      llvm::MemoryBufferRef(bytecode, "legacy-global-felt-initializers"), &upgradedBlock,
      ParserConfig(&ctx)
  )));
  ASSERT_EQ(upgradedBlock.getOperations().size(), 1u);
  auto upgraded = cast<ModuleOp>(upgradedBlock.front());
  EXPECT_TRUE(succeeded(mlir::verify(upgraded)));

  SmallVector<GlobalDefOp> globals;
  upgraded.walk([&](GlobalDefOp global) { globals.push_back(global); });
  ASSERT_EQ(globals.size(), 4u);

  FeltType expectedType = FeltType::get(&ctx, "bn128");
  for (GlobalDefOp global : globals) {
    if (auto arrayType = llvm::dyn_cast<llzk::array::ArrayType>(global.getType())) {
      EXPECT_EQ(arrayType.getElementType(), expectedType)
          << "array element type should agree with its initializer field";
      for (Attribute element : llvm::cast<ArrayAttr>(global.getInitialValueAttr())) {
        EXPECT_EQ(llvm::cast<FeltConstAttr>(element).getType(), expectedType);
      }
    } else {
      EXPECT_EQ(global.getType(), expectedType)
          << "scalar global should have adopted its initializer's field";
      EXPECT_EQ(llvm::cast<FeltConstAttr>(global.getInitialValueAttr()).getType(), expectedType);
    }
  }
}

} // namespace
