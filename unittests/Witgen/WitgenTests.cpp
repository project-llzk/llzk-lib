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
#include "WitgenDriver.h"
#include "WitgenUtils.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Field.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Parser/Parser.h>

#include <climits>
#include <limits>
#include <random>

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

TEST_F(WitgenTests, ParseJSONArrayNestedInput) {
  auto parsed = llvm::json::parse(R"([[1, 2], [3, "4"]])");
  ASSERT_TRUE(static_cast<bool>(parsed));

  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {2, 2});
  auto field = Field::tryGetField("babybear");
  ASSERT_TRUE(succeeded(field));

  auto value = witgen::parseJSONValue(&*parsed, arrayType, field->get(), nullptr);
  ASSERT_TRUE(static_cast<bool>(value)) << llvm::toString(value.takeError());

  auto arrayValue = std::get<witgen::ArrayValueRef>(*value);
  ASSERT_EQ(arrayValue->elements.size(), 4u);
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[0]), field->get().reduce(1));
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[1]), field->get().reduce(2));
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[2]), field->get().reduce(3));
  EXPECT_EQ(std::get<llvm::DynamicAPInt>(arrayValue->elements[3]), field->get().reduce(4));
}

TEST_F(WitgenTests, DefaultValueFailsOnUninitializedRead) {
  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto field = Field::tryGetField("babybear");
  ASSERT_TRUE(succeeded(field));

  SymbolTableCollection tables;
  std::mt19937_64 rng(0);
  auto value = witgen::defaultValue(
      feltType, tables, nullptr, field->get(), witgen::UninitializedBehavior::Fail, &rng
  );
  ASSERT_TRUE(static_cast<bool>(value));
  auto felt = witgen::asFelt(*value);
  ASSERT_FALSE(static_cast<bool>(felt));
  EXPECT_NE(llvm::toString(felt.takeError()).find("uninitialized felt value"), std::string::npos);
}

TEST_F(WitgenTests, RandomDefaultValueIsSeeded) {
  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {2});
  auto field = Field::tryGetField("babybear");
  ASSERT_TRUE(succeeded(field));

  SymbolTableCollection tables;
  std::mt19937_64 rngA(1234);
  std::mt19937_64 rngB(1234);
  auto lhs = witgen::defaultValue(
      arrayType, tables, nullptr, field->get(), witgen::UninitializedBehavior::Random, &rngA
  );
  auto rhs = witgen::defaultValue(
      arrayType, tables, nullptr, field->get(), witgen::UninitializedBehavior::Random, &rngB
  );
  ASSERT_TRUE(static_cast<bool>(lhs));
  ASSERT_TRUE(static_cast<bool>(rhs));

  auto lhsArray = std::get<witgen::ArrayValueRef>(*lhs);
  auto rhsArray = std::get<witgen::ArrayValueRef>(*rhs);
  ASSERT_EQ(lhsArray->elements.size(), rhsArray->elements.size());
  ASSERT_EQ(lhsArray->elements.size(), 2u);
  EXPECT_EQ(
      std::get<llvm::DynamicAPInt>(lhsArray->elements[0]),
      std::get<llvm::DynamicAPInt>(rhsArray->elements[0])
  );
  EXPECT_EQ(
      std::get<llvm::DynamicAPInt>(lhsArray->elements[1]),
      std::get<llvm::DynamicAPInt>(rhsArray->elements[1])
  );
}

TEST_F(WitgenTests, SharedWitgenRngUsesDeterministicSeed) {
  witgen::WitgenOptions options;
  options.randomSeed = 1234;

  std::mt19937_64 lhs = witgen::makeDefaultValueRng(options);
  std::mt19937_64 rhs = witgen::makeDefaultValueRng(options);

  EXPECT_EQ(lhs(), rhs());
  EXPECT_EQ(lhs(), rhs());
}

TEST_F(WitgenTests, ParseJSONArrayRejectsDynamicShape) {
  auto parsed = llvm::json::parse(R"([1])");
  ASSERT_TRUE(static_cast<bool>(parsed));

  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {ShapedType::kDynamic});
  auto field = Field::tryGetField("babybear");
  ASSERT_TRUE(succeeded(field));

  auto value = witgen::parseJSONValue(&*parsed, arrayType, field->get(), nullptr);
  ASSERT_FALSE(static_cast<bool>(value));
  EXPECT_NE(llvm::toString(value.takeError()).find("requires a static shape"), std::string::npos);
}

TEST_F(WitgenTests, SerializeJSONArrayRejectsDynamicShape) {
  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {ShapedType::kDynamic});

  auto arrayValue = std::make_shared<witgen::ArrayValue>();
  arrayValue->type = arrayType;
  arrayValue->elements.push_back(Field::getField("babybear").reduce(1));

  SymbolTableCollection tables;
  auto value = witgen::serializeJSONValue(arrayValue, arrayType, tables, nullptr);
  ASSERT_FALSE(static_cast<bool>(value));
  EXPECT_NE(llvm::toString(value.takeError()).find("requires a static shape"), std::string::npos);
}

TEST_F(WitgenTests, SerializeJSONArrayNestedOutput) {
  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {2, 2});

  auto arrayValue = std::make_shared<witgen::ArrayValue>();
  arrayValue->type = arrayType;
  auto field = Field::getField("babybear");
  arrayValue->elements.push_back(field.reduce(1));
  arrayValue->elements.push_back(field.reduce(2));
  arrayValue->elements.push_back(field.reduce(3));
  arrayValue->elements.push_back(field.reduce(4));

  SymbolTableCollection tables;
  auto value = witgen::serializeJSONValue(arrayValue, arrayType, tables, nullptr);
  ASSERT_TRUE(static_cast<bool>(value)) << llvm::toString(value.takeError());

  auto *outer = value->getAsArray();
  ASSERT_NE(outer, nullptr);
  ASSERT_EQ(outer->size(), 2u);
  auto *first = (*outer)[0].getAsArray();
  auto *second = (*outer)[1].getAsArray();
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(first->size(), 2u);
  ASSERT_EQ(second->size(), 2u);
  EXPECT_EQ((*first)[0].getAsString(), "1");
  EXPECT_EQ((*first)[1].getAsString(), "2");
  EXPECT_EQ((*second)[0].getAsString(), "3");
  EXPECT_EQ((*second)[1].getAsString(), "4");
}

TEST_F(WitgenTests, CheckedDynamicAPIntToSizeRejectsOverflow) {
  llvm::DynamicAPInt value = llzk::toDynamicAPInt(
      llvm::APInt(sizeof(size_t) * CHAR_BIT + 1, std::numeric_limits<size_t>::max())
  );
  value += llvm::DynamicAPInt(1);
  auto checked = witgen::checkedDynamicAPIntToSize(value, "unit-test overflow");
  ASSERT_FALSE(static_cast<bool>(checked));
  EXPECT_NE(llvm::toString(checked.takeError()).find("overflow"), std::string::npos);
}

TEST_F(WitgenTests, NestedAggregateFailModeMaterializesMonostate) {
  auto feltType = felt::FeltType::get(&ctx, StringAttr::get(&ctx, "babybear"));
  auto arrayType = array::ArrayType::get(feltType, {2});
  auto field = Field::tryGetField("babybear");
  ASSERT_TRUE(succeeded(field));

  SymbolTableCollection tables;
  std::mt19937_64 rng(0);
  auto value = witgen::defaultValue(
      arrayType, tables, nullptr, field->get(), witgen::UninitializedBehavior::Fail, &rng
  );
  ASSERT_TRUE(static_cast<bool>(value));
  auto arrayValue = std::get<witgen::ArrayValueRef>(*value);
  ASSERT_EQ(arrayValue->elements.size(), 2u);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(arrayValue->elements[0]));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(arrayValue->elements[1]));
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

TEST_F(WitgenTests, InterpreterHandlesNegativeUnsignedDivUIOperands) {
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
  witgen::FunctionInterpreter interpreter(
      *module, tables, Field::getField("babybear"), witgen::UninitializedBehavior::Zero,
      std::mt19937_64(0)
  );
  llvm::SmallVector<witgen::WitnessVal> args = {int64_t(-1), int64_t(2)};
  auto results = interpreter.run(func, args);
  ASSERT_TRUE(static_cast<bool>(results)) << llvm::toString(results.takeError());
  ASSERT_EQ(results->size(), 1u);
  auto quotient = witgen::asIndex((*results)[0]);
  ASSERT_TRUE(static_cast<bool>(quotient)) << llvm::toString(quotient.takeError());
  EXPECT_EQ(*quotient, std::numeric_limits<int64_t>::max());
}

TEST_F(WitgenTests, InterpreterHandlesNegativeUnsignedForBounds) {
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
  witgen::FunctionInterpreter interpreter(
      *module, tables, Field::getField("babybear"), witgen::UninitializedBehavior::Zero,
      std::mt19937_64(0)
  );
  llvm::SmallVector<witgen::WitnessVal> args = {int64_t(-1), int64_t(1), int64_t(2)};
  auto results = interpreter.run(func, args);
  ASSERT_TRUE(static_cast<bool>(results)) << llvm::toString(results.takeError());
  ASSERT_EQ(results->size(), 1u);
  auto count = witgen::asIndex((*results)[0]);
  ASSERT_TRUE(static_cast<bool>(count)) << llvm::toString(count.takeError());
  EXPECT_EQ(*count, 0);
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
  witgen::FunctionInterpreter interpreter(
      *module, tables, field, witgen::UninitializedBehavior::Zero, std::mt19937_64(0)
  );
  llvm::SmallVector<witgen::WitnessVal> args = {overflowingValue};
  auto results = interpreter.run(func, args);
  llvm::Error error = results.takeError();
  ASSERT_TRUE(static_cast<bool>(error))
      << "expected interpreter to reject unsigned-to-signed index underflow";
  EXPECT_NE(llvm::toString(std::move(error)).find("fit in index"), std::string::npos);
}

TEST_F(WitgenTests, InterpreterFailsDuringFinalWitnessSerialization) {
  constexpr llvm::StringLiteral source = R"mlir(
    module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {
      struct.def @Main {
        struct.member @out : !felt.type<"babybear"> {llzk.pub}
        function.def @compute() -> !struct.type<@Main> {
          %self = struct.new : !struct.type<@Main>
          %x = llzk.nondet : !felt.type<"babybear">
          struct.writem %self[@out] = %x : !struct.type<@Main>, !felt.type<"babybear">
          function.return %self : !struct.type<@Main>
        }
        function.def @constrain(%self: !struct.type<@Main>) {
          function.return
        }
      }
    }
  )mlir";

  auto module = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(static_cast<bool>(module));

  SymbolTableCollection tables;
  witgen::Interpreter interpreter(
      *module, tables, Field::getField("babybear"), witgen::UninitializedBehavior::Fail,
      std::mt19937_64(0)
  );
  interpreter.setOutputScope(witgen::OutputScope::Public);
  auto result = interpreter.runMainFromJSON(llvm::json::Object {});
  ASSERT_FALSE(static_cast<bool>(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("uninitialized felt value"), std::string::npos);
}
