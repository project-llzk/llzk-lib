//===-- WitgenTests.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"
#include "JSON.h"
#include "ValueModel.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;
using namespace llzk;

class WitgenTests : public LLZKTest {};

TEST_F(WitgenTests, ParseJSONArrayInput) {
  auto parsed = llvm::json::parse(R"([1, "2", 3])");
  ASSERT_TRUE(static_cast<bool>(parsed));

  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {3});
  auto field = Field::tryGetField("babybear");
  ASSERT_TRUE(succeeded(field));

  auto value = witgen::parseJSONValue(&*parsed, arrayType, field->get(), nullptr);
  ASSERT_TRUE(static_cast<bool>(value)) << llvm::toString(value.takeError());

  auto arrayValue = std::get<witgen::ArrayValueRef>(*value);
  ASSERT_EQ(arrayValue->elements.size(), 3u);
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[0]), field->get().reduce(1));
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[1]), field->get().reduce(2));
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[2]), field->get().reduce(3));
}

TEST_F(WitgenTests, SerializeStructOnlyEmitsPublicMembers) {
  constexpr llvm::StringLiteral source = R"llzk(
    module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {
      struct.def @Main {
        struct.member @out : !felt.type<"babybear"> {llzk.pub}
        struct.member @tmp : !felt.type<"babybear">
        function.def @compute() -> !struct.type<@Main> {
          %self = struct.new : !struct.type<@Main>
          function.return %self : !struct.type<@Main>
        }
        function.def @constrain(%self: !struct.type<@Main>) {
          function.return
        }
      }
    }
  )llzk";

  auto module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(static_cast<bool>(module));

  SymbolTableCollection tables;
  auto mainType = getMainInstanceType(module->getOperation());
  ASSERT_TRUE(succeeded(mainType));
  ASSERT_TRUE(static_cast<bool>(mainType.value()));

  auto structValue = std::make_shared<witgen::StructValue>();
  structValue->type = *mainType;
  structValue->members["out"] = Field::getField("babybear").reduce(7);
  structValue->members["tmp"] = Field::getField("babybear").reduce(9);

  auto json = witgen::serializeJSONValue(structValue, *mainType, tables, module->getOperation());
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());

  auto *obj = json->getAsObject();
  ASSERT_NE(obj, nullptr);
  EXPECT_NE(obj->get("out"), nullptr);
  EXPECT_EQ(obj->get("tmp"), nullptr);
}
