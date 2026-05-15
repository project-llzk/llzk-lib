//===-- WitgenTests.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"
#include "Interpreter.h"
#include "JSON.h"
#include "ValueModel.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <limits>

using namespace mlir;
using namespace llzk;

class WitgenTests : public LLZKTest {};

static function::FuncDefOp getUniqueFuncByName(ModuleOp module, StringRef name) {
  function::FuncDefOp result;
  module.walk([&](function::FuncDefOp funcOp) {
    if (funcOp.getSymName() == name) {
      result = funcOp;
    }
  });
  return result;
}

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

TEST_F(WitgenTests, InterpreterRejectsNegativeUnsignedDivUIOperands) {
  constexpr llvm::StringLiteral source = R"mlir(
    module attributes {llzk.lang} {
      function.def @u_div(%lhs: index, %rhs: index) -> index {
        %q = arith.divui %lhs, %rhs : index
        function.return %q : index
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(static_cast<bool>(module));

  auto func = getUniqueFuncByName(*module, "u_div");
  ASSERT_TRUE(static_cast<bool>(func));

  SymbolTableCollection tables;
  witgen::FunctionInterpreter interpreter(*module, tables, Field::getField("babybear"));
  llvm::SmallVector<witgen::WitnessVal> args = {int64_t(-1), int64_t(2)};
  auto results = interpreter.run(func, args);
  llvm::Error error = results.takeError();
  ASSERT_TRUE(static_cast<bool>(error)) << "expected interpreter to reject negative divui operand";
  EXPECT_NE(
      llvm::toString(std::move(error))
          .find("cannot reinterpret a negative index value as unsigned"),
      std::string::npos
  );
}

TEST_F(WitgenTests, InterpreterRejectsNegativeUnsignedForBounds) {
  constexpr llvm::StringLiteral source = R"mlir(
    module attributes {llzk.lang} {
      function.def @count_unsigned(%lb: index, %ub: index, %step: index) -> index {
        %c0 = arith.constant 0 : index
        %count = scf.for %iv = %lb to %ub step %step iter_args(%acc = %c0) -> (index) {
          %c1 = arith.constant 1 : index
          %next = arith.addi %acc, %c1 : index
          scf.yield %next : index
        }
        function.return %count : index
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(static_cast<bool>(module));

  scf::ForOp forOp;
  module->walk([&](scf::ForOp op) { forOp = op; });
  ASSERT_TRUE(static_cast<bool>(forOp));
  forOp->setAttr("unsignedCmp", UnitAttr::get(&ctx));

  auto func = getUniqueFuncByName(*module, "count_unsigned");
  ASSERT_TRUE(static_cast<bool>(func));

  SymbolTableCollection tables;
  witgen::FunctionInterpreter interpreter(*module, tables, Field::getField("babybear"));
  llvm::SmallVector<witgen::WitnessVal> args = {int64_t(-1), int64_t(1), int64_t(2)};
  auto results = interpreter.run(func, args);
  llvm::Error error = results.takeError();
  ASSERT_TRUE(static_cast<bool>(error))
      << "expected interpreter to reject negative unsigned loop bound";
  EXPECT_NE(
      llvm::toString(std::move(error))
          .find("cannot reinterpret a negative index value as unsigned"),
      std::string::npos
  );
}

TEST_F(WitgenTests, InterpreterRejectsUnsignedToSignedIndexUnderflow) {
  auto field = Field::getField("goldilocks");
  auto overflowingValue = field.reduce(llvm::APInt(64, uint64_t(1) << 63));

  constexpr llvm::StringLiteral source = R"mlir(
    module attributes {llzk.lang} {
      function.def @felt_to_index(%value: !felt.type<"goldilocks">) -> index
          attributes {function.allow_non_native_field_ops} {
        %idx = cast.toindex %value : !felt.type<"goldilocks">
        function.return %idx : index
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(static_cast<bool>(module));

  auto func = getUniqueFuncByName(*module, "felt_to_index");
  ASSERT_TRUE(static_cast<bool>(func));

  SymbolTableCollection tables;
  witgen::FunctionInterpreter interpreter(*module, tables, field);
  llvm::SmallVector<witgen::WitnessVal> args = {overflowingValue};
  auto results = interpreter.run(func, args);
  llvm::Error error = results.takeError();
  ASSERT_TRUE(static_cast<bool>(error))
      << "expected interpreter to reject unsigned-to-signed index underflow";
  EXPECT_NE(
      llvm::toString(std::move(error)).find("unsigned value does not fit in signed int64_t"),
      std::string::npos
  );
}
