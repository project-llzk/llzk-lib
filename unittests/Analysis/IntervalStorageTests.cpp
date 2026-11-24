//===-- IntervalStorageTests.cpp - Interval storage regressions -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"
#include "../LLZKTestUtils.h"

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;

namespace {

/// Exercise stored intervals and symbolic identities through the public analysis API.
class IntervalStorageTests : public LLZKTest {
protected:
  const Field &field = Field::getField("babybear");

  OwningOpRef<ModuleOp> parseModule(llvm::StringRef source) {
    return parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  }

  /// Check asserted inequality or conflicting constants on the two returned reads.
  void checkReadConstraints(std::string source, bool constants, bool shouldAlias) {
    SCOPED_TRACE(constants ? "conflicting constants" : "inequality");
    source.replace(
        source.find("// CONSTRAINTS"), StringRef("// CONSTRAINTS").size(),
        constants ? R"mlir(
    %leftCheck = bool.cmp eq(%left, %seven)
    bool.assert %leftCheck
    %rightCheck = bool.cmp eq(%right, %nine)
    bool.assert %rightCheck
)mlir"
                  : R"mlir(
    %different = bool.cmp ne(%left, %right)
    bool.assert %different
)mlir"
    );
    auto mod = parseModule(source);
    ASSERT_TRUE(mod);
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    for (auto comparison : fn.getOps<boolean::CmpOp>()) {
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(comparison.getResult());
      ASSERT_NE(lattice, nullptr);
      ASSERT_FALSE(lattice->getConstraints().empty());
      for (const ExpressionValue &constraint : lattice->getConstraints()) {
        smt->addConstraint(constraint.getExpr());
      }
    }
    if (constants) {
      for (auto [i, value] : llvm::enumerate(fn.getBody().front().getTerminator()->getOperands())) {
        const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
        ASSERT_NE(lattice, nullptr);
        auto expected = shouldAlias ? Interval::Empty(field)
                                    : Interval::Degenerate(field, field.felt(i == 0 ? 7 : 9));
        EXPECT_EQ(lattice->getValue().getScalarValue().getInterval(), expected);
      }
    }
    EXPECT_EQ(smt->check(), std::optional<bool>(!shouldAlias));
  }
};

// A ranged write does not make independent selections equal to its stored expression.
TEST_F(IntervalStorageTests, RangedReadsKeepIndependentSymbols) {
  for (bool storedRange : {false, true}) {
    SCOPED_TRACE(storedRange ? "ranged writes" : "unwritten ranges");
    std::string source = R"mlir(
!P = !pod.type<[@value: !felt.type]>
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !felt.type>, %pods: !array.type<2 x !P>, %i: index, %j: index, %input: !felt.type) -> (!felt.type, !felt.type, !felt.type, !felt.type, !felt.type, !felt.type) {
    %zero = arith.constant 0 : index
    array.write %array[%zero] = %input : !array.type<2 x !felt.type>, !felt.type
    %leftPod = array.read %pods[%i] : !array.type<2 x !P>, !P
    %rightPod = array.read %pods[%j] : !array.type<2 x !P>, !P
    // WRITES
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %array[%j] : !array.type<2 x !felt.type>, !felt.type
    %nestedLeft = pod.read %leftPod[@value] : !P, !felt.type
    %nestedRight = pod.read %rightPod[@value] : !P, !felt.type
    %exactLeft = array.read %array[%zero] : !array.type<2 x !felt.type>, !felt.type
    %exactRight = array.read %array[%zero] : !array.type<2 x !felt.type>, !felt.type
    function.return %left, %right, %nestedLeft, %nestedRight, %exactLeft, %exactRight : !felt.type, !felt.type, !felt.type, !felt.type, !felt.type, !felt.type
  }
}
)mlir";
    if (storedRange) {
      source.replace(
          source.find("// WRITES"), std::string("// WRITES").size(),
          "array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type\n"
          "    pod.write %leftPod[@value] = %input : !P, !felt.type"
      );
    }
    auto mod = parseModule(source);
    ASSERT_TRUE(mod);
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);
    ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto returned = fn.getBody().front().getTerminator()->getOperands();
    SmallVector<llvm::SMTExprRef> expressions;
    for (Value value : returned) {
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
      ASSERT_NE(lattice, nullptr);
      const auto &stored = lattice->getValue().getScalarValue();
      EXPECT_EQ(stored.getInterval(), Interval::Entire(field));
      EXPECT_EQ(
          stored.getOptionalUnreducedInterval(), UnreducedInterval(field.zero(), field.maxVal())
      );
      expressions.push_back(stored.getExpr());
      ASSERT_NE(expressions.back(), nullptr);
    }
    for (unsigned first : {0U, 2U}) {
      EXPECT_FALSE(*expressions[first] == *expressions[first + 1]);
      EXPECT_EQ(expressions[first], analysis->getOrCreateSymbol(returned[first]));
      EXPECT_EQ(expressions[first + 1], analysis->getOrCreateSymbol(returned[first + 1]));
      smt->addConstraint(smt->mkEqual(
          expressions[first], smt->mkBitvector(toAPSInt(field.felt(7)), field.bitWidth())
      ));
      smt->addConstraint(smt->mkEqual(
          expressions[first + 1], smt->mkBitvector(toAPSInt(field.felt(9)), field.bitWidth())
      ));
    }
    EXPECT_EQ(smt->check(), std::optional<bool>(true));
    EXPECT_TRUE(*expressions[4] == *expressions[5]);
    EXPECT_EQ(expressions[4], analysis->getOrCreateSymbol(fn.getArgument(4)));
  }
}

// The same address does not imply the same value when a write separates the reads.
TEST_F(IntervalStorageTests, DynamicReadConstraintsDoNotCrossWrites) {
  auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !felt.type>, %i: index, %input: !felt.type) -> (!felt.type, !felt.type) {
    %seven = felt.const 7
    %nine = felt.const 9
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %leftCheck = bool.cmp eq(%left, %seven)
    bool.assert %leftCheck
    %rightCheck = bool.cmp eq(%right, %nine)
    bool.assert %rightCheck
    function.return %left, %right : !felt.type, !felt.type
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto smt = llvm::CreateZ3Solver();
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  llzk::dataflow::loadRequiredAnalyses(solver);
  solver.load<SourceRefAnalysis>();
  auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
  ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  for (auto [i, value] : llvm::enumerate(fn.getBody().front().getTerminator()->getOperands())) {
    const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
    ASSERT_NE(lattice, nullptr);
    const auto &actual = lattice->getValue().getScalarValue();
    auto expected = Interval::Degenerate(field, field.felt(i == 0 ? 7 : 9));
    EXPECT_TRUE(checkCond(expected, actual.getInterval(), expected == actual.getInterval()));
    EXPECT_EQ(actual.getExpr(), analysis->getOrCreateSymbol(value));
  }
}

// Relational constraints need equal solver values for proven aliases, even with entire bounds.
TEST_F(IntervalStorageTests, ProvenRangedAliasesHaveEqualSolverValues) {
  for (bool memberReads : {false, true}) {
    SCOPED_TRACE(memberReads ? "component members" : "array elements");
    for (unsigned selection : {0U, 1U, 2U, 3U}) {
      // Same index, independent indices, intervening writes, or one repeated SSA value.
      SCOPED_TRACE(selection);
      std::string source = R"mlir(
module attributes {llzk.lang} {
  struct.def @Cell {
    struct.member @value : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@Cell> {
      %self = struct.new : !struct.type<@Cell>
      function.return %self : !struct.type<@Cell>
    }
    function.def @constrain(%self: !struct.type<@Cell>) {
      function.return
    }
  }
  function.def @test(%array: !array.type<2 x !felt.type>, %cells: !array.type<2 x !struct.type<@Cell>>, %i: index, %j: index, %input: !felt.type, %replacement: !struct.type<@Cell>) {
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    // ARRAY_WRITE
    %right = array.read %array[%j] : !array.type<2 x !felt.type>, !felt.type
    %leftCell = array.read %cells[%i] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    // COMPONENT_WRITE
    %rightCell = array.read %cells[%j] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %leftMember = struct.readm %leftCell[@value] : !struct.type<@Cell>, !felt.type
    %rightMember = struct.readm %rightCell[@value] : !struct.type<@Cell>, !felt.type
    %different = bool.cmp ne(%left, %right)
    // Revisit both reads in opposite directions without changing their intervals.
    bool.assert %different
    bool.assert %different
    bool.assert %different
    function.return
  }
}
)mlir";
      if (selection == 2) {
        source.replace(
            source.find("// ARRAY_WRITE"), std::string("// ARRAY_WRITE").size(),
            "array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type"
        );
        source.replace(
            source.find("// COMPONENT_WRITE"), std::string("// COMPONENT_WRITE").size(),
            "array.write %cells[%i] = %replacement : !array.type<2 x !struct.type<@Cell>>, "
            "!struct.type<@Cell>"
        );
      }
      auto mod = parseModule(source);
      ASSERT_TRUE(mod);
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      if (selection != 1) {
        for (auto read : fn.getOps<array::ReadArrayOp>()) {
          read->setOperand(1, fn.getArgument(2));
        }
      }
      auto comparison = *fn.getOps<boolean::CmpOp>().begin();
      if (memberReads) {
        auto members = llvm::to_vector(fn.getOps<component::MemberReadOp>());
        comparison->setOperand(0, members[0]->getResult(0));
        comparison->setOperand(1, members[1]->getResult(0));
      }
      if (selection == 3) {
        comparison->setOperand(1, comparison.getLhs());
      }
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(comparison.getResult());
      ASSERT_NE(lattice, nullptr);
      ASSERT_FALSE(lattice->getConstraints().empty());
      if (selection == 0) {
        for (Value read : comparison->getOperands()) {
          const auto *readLattice = solver.lookupState<IntervalAnalysisLattice>(read);
          ASSERT_NE(readLattice, nullptr);
          EXPECT_EQ(
              readLattice->getValue().getScalarValue().getInterval(), Interval::Entire(field)
          );
        }
      }
      // Inspect before check(), which may simplify duplicate solver assertions.
      std::string assertions = buildStringViaPrint(*smt);
      EXPECT_EQ(StringRef(assertions).count("(assert "), selection == 0 ? 1U : 0U) << assertions;
      for (const ExpressionValue &constraint : lattice->getConstraints()) {
        smt->addConstraint(constraint.getExpr());
      }
      EXPECT_EQ(smt->check(), std::optional<bool>(selection == 1 || selection == 2));
    }
  }
}

// Missing read expressions defer only the SMT equality, which a later visit can still assert.
TEST_F(IntervalStorageTests, RangedAliasesWithMissingExpressionsCanBeRevisited) {
  for (unsigned initializedReads : {0U, 1U, 2U}) {
    // Neither expression initialized, only the left initialized, or only the right initialized.
    SCOPED_TRACE(initializedReads);
    auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%i: index, %bound: !felt.type) {
    %array = "builtin.unrealized_conversion_cast"() : () -> !array.type<2 x !felt.type>
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %leftCheck = bool.cmp eq(%left, %bound)
    bool.assert %leftCheck
    %rightCheck = bool.cmp eq(%right, %right)
    bool.assert %rightCheck
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto reads = llvm::to_vector(fn.getOps<array::ReadArrayOp>());
    auto asserts = llvm::to_vector(fn.getOps<boolean::AssertOp>());
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    SmallVector<IntervalAnalysisLattice *> readLattices;
    for (auto [i, read] : llvm::enumerate(reads)) {
      auto *lattice = solver.getOrCreateState<IntervalAnalysisLattice>(read.getResult());
      ExpressionValue value(field);
      if (initializedReads & (1U << i)) {
        value = value.withExpression(analysis->getOrCreateSymbol(read.getResult()));
      }
      (void)lattice->setValue(value);
      readLattices.push_back(lattice);
    }
    ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
    EXPECT_EQ(StringRef(buildStringViaPrint(*smt)).count("(assert "), 0U);
    for (auto [i, lattice] : llvm::enumerate(readLattices)) {
      const auto &value = lattice->getValue().getScalarValue();
      EXPECT_EQ(value.getExpr() != nullptr, (initializedReads & (1U << i)) != 0);
      EXPECT_EQ(value.getInterval(), Interval::Entire(field));
    }

    // Replay an assertion on the live worklist without reinitializing the analysis states.
    // This deliberately tests explicit revisits, not automatic retry of a deferred equality.
    auto revisit = [&](boolean::AssertOp assertOp) {
      solver.enqueue({solver.getProgramPointAfter(assertOp), analysis});
      return solver.initializeAndRun(*mod, [](DataFlowAnalysis &) { return false; });
    };

    // Narrow one read while expressions are still missing. Its alias must narrow as well.
    Interval narrowed = UnreducedInterval(0, 7).reduce(field);
    auto *boundLattice = solver.getOrCreateState<IntervalAnalysisLattice>(fn.getArgument(1));
    (void)boundLattice->setValue(
        ExpressionValue(analysis->getOrCreateSymbol(fn.getArgument(1)), narrowed)
    );
    ASSERT_TRUE(succeeded(revisit(asserts.front())));
    EXPECT_EQ(StringRef(buildStringViaPrint(*smt)).count("(assert "), 0U);
    for (auto [i, lattice] : llvm::enumerate(readLattices)) {
      const auto &value = lattice->getValue().getScalarValue();
      EXPECT_EQ(value.getExpr() != nullptr, (initializedReads & (1U << i)) != 0);
      EXPECT_EQ(value.getInterval(), narrowed);
    }

    // Initialize the missing symbols without changing the bounds, then revisit repeatedly.
    for (auto [read, lattice] : llvm::zip(reads, readLattices)) {
      (void)lattice->setValue(lattice->getValue().getScalarValue().withExpression(
          analysis->getOrCreateSymbol(read.getResult())
      ));
    }
    for (unsigned visit = 0; visit < 3; ++visit) {
      ASSERT_TRUE(succeeded(revisit(asserts.back())));
      EXPECT_EQ(StringRef(buildStringViaPrint(*smt)).count("(assert "), 1U);
      for (auto *lattice : readLattices) {
        EXPECT_EQ(lattice->getValue().getScalarValue().getInterval(), narrowed);
      }
    }

    // Once equality is cached, further interval restrictions must still reach both aliases.
    Interval singleton = Interval::Degenerate(field, field.felt(3));
    (void)boundLattice->setValue(boundLattice->getValue().getScalarValue().withInterval(singleton));
    ASSERT_TRUE(succeeded(revisit(asserts.front())));
    EXPECT_EQ(StringRef(buildStringViaPrint(*smt)).count("(assert "), 1U);
    for (auto [read, lattice] : llvm::zip(reads, readLattices)) {
      const auto &value = lattice->getValue().getScalarValue();
      EXPECT_EQ(value.getInterval(), singleton);
      EXPECT_EQ(value.getExpr(), analysis->getOrCreateSymbol(read.getResult()));
    }
    smt->addConstraint(smt->mkNot(smt->mkEqual(
        readLattices.front()->getValue().getScalarValue().getExpr(),
        readLattices.back()->getValue().getScalarValue().getExpr()
    )));
    EXPECT_EQ(smt->check(), std::optional<bool>(false));
  }
}

// Unrelated array reads cannot change repeated reads of the same dynamic selection.
TEST_F(IntervalStorageTests, InterveningArrayReadsPreserveDynamicAliases) {
  for (bool memberReads : {false, true}) {
    SCOPED_TRACE(memberReads ? "component members" : "array elements");
    for (bool relational : {false, true}) {
      SCOPED_TRACE(relational ? "inequality" : "conflicting constants");
      std::string source = R"mlir(
module attributes {llzk.lang} {
  struct.def @Cell {
    struct.member @value : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@Cell> {
      %self = struct.new : !struct.type<@Cell>
      function.return %self : !struct.type<@Cell>
    }
    function.def @constrain(%self: !struct.type<@Cell>) {
      function.return
    }
  }
  function.def @test(%array: !array.type<2 x !felt.type>, %cells: !array.type<2 x !struct.type<@Cell>>, %other: !array.type<2 x !felt.type>, %i: index, %j: index) {
    %seven = felt.const 7
    %nine = felt.const 9
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %betweenElements = array.read %other[%j] : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %leftCell = array.read %cells[%i] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %betweenCells = array.read %other[%j] : !array.type<2 x !felt.type>, !felt.type
    %rightCell = array.read %cells[%i] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %leftMember = struct.readm %leftCell[@value] : !struct.type<@Cell>, !felt.type
    %betweenMembers = array.read %other[%j] : !array.type<2 x !felt.type>, !felt.type
    %rightMember = struct.readm %rightCell[@value] : !struct.type<@Cell>, !felt.type
    // CONSTRAINTS
    function.return
  }
}
)mlir";
      source.replace(
          source.find("// CONSTRAINTS"), std::string("// CONSTRAINTS").size(),
          relational ? R"mlir(
    %different = bool.cmp ne(%left, %right)
    bool.assert %different
)mlir"
                     : R"mlir(
    %leftCheck = bool.cmp eq(%left, %seven)
    bool.assert %leftCheck
    %rightCheck = bool.cmp eq(%right, %nine)
    bool.assert %rightCheck
)mlir"
      );
      auto mod = parseModule(source);
      ASSERT_TRUE(mod);
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      auto comparisons = llvm::to_vector(fn.getOps<boolean::CmpOp>());
      if (memberReads) {
        auto members = llvm::to_vector(fn.getOps<component::MemberReadOp>());
        comparisons[0]->setOperand(0, members[0]->getResult(0));
        if (relational) {
          comparisons[0]->setOperand(1, members[1]->getResult(0));
        } else {
          comparisons[1]->setOperand(0, members[1]->getResult(0));
        }
      }
      auto smt = llvm::CreateZ3Solver();
      DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
      llzk::dataflow::loadRequiredAnalyses(solver);
      solver.load<SourceRefAnalysis>();
      solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
      ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
      for (auto comparison : comparisons) {
        const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(comparison.getResult());
        ASSERT_NE(lattice, nullptr);
        ASSERT_FALSE(lattice->getConstraints().empty());
        for (const ExpressionValue &constraint : lattice->getConstraints()) {
          smt->addConstraint(constraint.getExpr());
        }
        if (!relational) {
          const auto *read = solver.lookupState<IntervalAnalysisLattice>(comparison.getLhs());
          ASSERT_NE(read, nullptr);
          EXPECT_EQ(read->getValue().getScalarValue().getInterval(), Interval::Empty(field));
        }
      }
      EXPECT_EQ(smt->check(), std::optional<bool>(false));
    }
  }
}

// Constraints on separate dynamic selections must not flow through equal-looking ranges.
TEST_F(IntervalStorageTests, RangedReadConstraintsStayIndependent) {
  for (unsigned selection : {0U, 1U, 2U}) {
    SCOPED_TRACE(selection); // Independent indices, same concrete index, same SSA index.
    auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  struct.def @Cell {
    struct.member @value : !felt.type {llzk.pub}
    function.def @compute() -> !struct.type<@Cell> {
      %self = struct.new : !struct.type<@Cell>
      function.return %self : !struct.type<@Cell>
    }
    function.def @constrain(%self: !struct.type<@Cell>) {
      function.return
    }
  }
  function.def @test(%array: !array.type<2 x !felt.type>, %cells: !array.type<2 x !struct.type<@Cell>>, %i: index, %j: index) -> (!felt.type, !felt.type, !felt.type, !felt.type) attributes {function.allow_constraint} {
    %zero = arith.constant 0 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %array[%j] : !array.type<2 x !felt.type>, !felt.type
    %leftCell = array.read %cells[%i] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %rightCell = array.read %cells[%j] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %leftMember = struct.readm %leftCell[@value] : !struct.type<@Cell>, !felt.type
    %rightMember = struct.readm %rightCell[@value] : !struct.type<@Cell>, !felt.type
    // Register aliases with entire intervals before narrowing either read.
    constrain.eq %left, %left : !felt.type
    constrain.eq %right, %right : !felt.type
    constrain.eq %leftMember, %leftMember : !felt.type
    constrain.eq %rightMember, %rightMember : !felt.type
    %leftCheck = bool.cmp eq(%left, %seven)
    bool.assert %leftCheck
    %rightCheck = bool.cmp eq(%right, %nine)
    bool.assert %rightCheck
    %leftMemberCheck = bool.cmp eq(%leftMember, %seven)
    bool.assert %leftMemberCheck
    %rightMemberCheck = bool.cmp eq(%rightMember, %nine)
    bool.assert %rightMemberCheck
    function.return %left, %right, %leftMember, %rightMember : !felt.type, !felt.type, !felt.type, !felt.type
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    if (selection != 0) {
      Value index = fn.getArgument(2);
      if (selection == 1) {
        index = fn.getBody().front().front().getResult(0);
      }
      for (auto read : fn.getOps<array::ReadArrayOp>()) {
        read->setOperand(1, index);
      }
    }
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
    for (auto [i, value] : llvm::enumerate(fn.getBody().front().getTerminator()->getOperands())) {
      SCOPED_TRACE(i);
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
      ASSERT_NE(lattice, nullptr);
      const auto &actual = lattice->getValue().getScalarValue().getInterval();
      auto expected = selection != 0 ? Interval::Empty(field)
                                     : Interval::Degenerate(field, field.felt(i % 2 == 0 ? 7 : 9));
      EXPECT_TRUE(checkCond(expected, actual, expected == actual));
      if (selection != 1) {
        EXPECT_EQ(
            lattice->getValue().getScalarValue().getExpr(), analysis->getOrCreateSymbol(value)
        );
      }
    }
  }
}

// Emitting equality or containment constraints cannot change a previously read element.
TEST_F(IntervalStorageTests, ConstraintEmissionsPreserveReadAliases) {
  for (bool containment : {false, true}) {
    SCOPED_TRACE(containment ? "constrain.in" : "constrain.eq");
    for (unsigned selection : {0U, 1U, 2U}) {
      SCOPED_TRACE(selection); // Same index, independent indices, or an intervening write.
      std::string source = R"mlir(
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !felt.type>, %i: index, %j: index, %input: !felt.type) -> (!felt.type, !felt.type) attributes {function.allow_constraint} {
    %seven = felt.const 7
    %nine = felt.const 9
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
    // EMISSION
    // WRITE
    %right = array.read %array[RIGHT_INDEX] : !array.type<2 x !felt.type>, !felt.type
    // CONSTRAINTS
    function.return %left, %right : !felt.type, !felt.type
  }
}
)mlir";
      source.replace(
          source.find("// EMISSION"), StringRef("// EMISSION").size(),
          containment ? "constrain.in %array, %left : !array.type<2 x !felt.type>, !felt.type"
                      : "constrain.eq %left, %seven : !felt.type"
      );
      source.replace(
          source.find("RIGHT_INDEX"), StringRef("RIGHT_INDEX").size(), selection == 1 ? "%j" : "%i"
      );
      if (selection == 2) {
        source.replace(
            source.find("// WRITE"), StringRef("// WRITE").size(),
            "array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type"
        );
      }
      for (bool constants : {false, true}) {
        checkReadConstraints(source, constants, selection == 0);
      }
    }
  }
}

// Equal extractions retain equal contents only until a write changes either observed copy.
TEST_F(IntervalStorageTests, EquivalentSubarrayReadsShareConstraints) {
  for (unsigned selection : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U}) {
    // Equal extractions, one row SSA value, independent row/column indices, or writes
    // between extractions, before both scalar reads (to either copy), or between scalar reads.
    SCOPED_TRACE(selection);
    std::string source = R"mlir(
module attributes {llzk.lang} {
  function.def @test(%matrix: !array.type<2,2 x !felt.type>, %i: index, %j: index, %k: index, %input: !felt.type) -> (!felt.type, !felt.type) {
    %seven = felt.const 7
    %nine = felt.const 9
    %row1 = array.extract %matrix[%i] : !array.type<2,2 x !felt.type>
    // BETWEEN_COPIES
    %row2 = array.extract %matrix[ROW_INDEX] : !array.type<2,2 x !felt.type>
    // BEFORE_READS
    %left = array.read %row1[%j] : !array.type<2 x !felt.type>, !felt.type
    // BETWEEN_READS
    %right = array.read ROW_VALUE[COLUMN_INDEX] : !array.type<2 x !felt.type>, !felt.type
    // CONSTRAINTS
    function.return %left, %right : !felt.type, !felt.type
  }
}
)mlir";
    source.replace(
        source.find("ROW_INDEX"), StringRef("ROW_INDEX").size(), selection == 2 ? "%k" : "%i"
    );
    source.replace(
        source.find("ROW_VALUE"), StringRef("ROW_VALUE").size(), selection == 1 ? "%row1" : "%row2"
    );
    source.replace(
        source.find("COLUMN_INDEX"), StringRef("COLUMN_INDEX").size(), selection == 3 ? "%k" : "%j"
    );
    if (selection == 4) {
      source.replace(
          source.find("// BETWEEN_COPIES"), StringRef("// BETWEEN_COPIES").size(),
          "array.write %matrix[%i, %j] = %input : !array.type<2,2 x !felt.type>, !felt.type"
      );
    } else if (selection >= 5) {
      StringRef location = selection == 7 ? "// BETWEEN_READS" : "// BEFORE_READS";
      source.replace(
          source.find(location.str()), location.size(),
          selection == 5
              ? "array.write %row1[%j] = %input : !array.type<2 x !felt.type>, !felt.type"
              : "array.write %row2[%j] = %input : !array.type<2 x !felt.type>, !felt.type"
      );
    }
    for (bool constants : {false, true}) {
      checkReadConstraints(source, constants, selection < 2);
    }
  }
}

} // namespace
