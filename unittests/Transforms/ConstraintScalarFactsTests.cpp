//===-- ConstraintScalarFactsTests.cpp - Scalar fact tests ------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../../lib/Dialect/Polymorphic/Transforms/ConstraintScalarFacts.h"
#include "../LLZKTestBase.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace {

/// Count evaluator visits without depending on wall-clock performance.
class CountedIdentityOp : public mlir::Op<
                              CountedIdentityOp, mlir::OpTrait::OneOperand,
                              mlir::OpTrait::OneResult, mlir::MemoryEffectOpInterface::Trait> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CountedIdentityOp)
  using Op::Op;
  inline static unsigned foldCalls = 0;

  static llvm::StringRef getOperationName() { return "fact_test.identity"; }
  static llvm::ArrayRef<llvm::StringRef> getAttributeNames() { return {}; }

  void getEffects(llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &) {}

  mlir::OpFoldResult fold(llvm::ArrayRef<mlir::Attribute> operands) {
    ++foldCalls;
    return operands.front();
  }
};

/// Register the counted operation used to inspect fact evaluation work.
class FactTestDialect : public mlir::Dialect {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FactTestDialect)
  explicit FactTestDialect(mlir::MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, mlir::TypeID::get<FactTestDialect>()) {
    addOperations<CountedIdentityOp>();
  }
  static llvm::StringRef getDialectNamespace() { return "fact_test"; }
};

class ConstraintScalarFactsTests : public LLZKTest {
protected:
  ConstraintScalarFactsTests() {
    ctx.getOrLoadDialect<FactTestDialect>();
    CountedIdentityOp::foldCalls = 0;
  }

  /// Build a fresh SSA result whose fold hook records each evaluation attempt.
  mlir::Value identity(mlir::OpBuilder &builder, mlir::Value value) {
    mlir::OperationState state(loc, CountedIdentityOp::getOperationName());
    state.addOperands(value);
    state.addTypes(value.getType());
    return builder.create(state)->getResult(0);
  }
};

TEST_F(ConstraintScalarFactsTests, UnresolvedChainIsEvaluatedOncePerOperation) {
  mlir::Block body;
  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&body);
  mlir::Value value = body.addArgument(builder.getIndexType(), loc);
  llzk::polymorphic::detail::ConstraintScalarFacts facts(&ctx);

  constexpr unsigned length = 256;
  for (unsigned i = 0; i < length; ++i) {
    value = identity(builder, value);
    EXPECT_FALSE(facts.known(value));
  }
  EXPECT_EQ(CountedIdentityOp::foldCalls, length);
  for (unsigned i = 0; i < 16; ++i) {
    EXPECT_FALSE(facts.known(value));
  }
  EXPECT_EQ(CountedIdentityOp::foldCalls, length);
}

TEST_F(ConstraintScalarFactsTests, UnresolvedResultsDoNotPoisonFreshConstantBindings) {
  mlir::Block body;
  mlir::OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&body);
  auto unknown = body.addArgument(builder.getIndexType(), loc);
  llzk::polymorphic::detail::ConstraintScalarFacts facts(&ctx);
  EXPECT_FALSE(facts.known(identity(builder, unknown)));

  for (int64_t i = 0; i < 8; ++i) {
    mlir::Value constant = mlir::arith::ConstantIndexOp::create(builder, loc, i);
    auto result = identity(builder, constant);
    EXPECT_EQ(facts.known(result), builder.getIndexAttr(i));
    EXPECT_EQ(facts.known(result), builder.getIndexAttr(i));
  }
  EXPECT_EQ(CountedIdentityOp::foldCalls, 9U);
}

} // namespace
