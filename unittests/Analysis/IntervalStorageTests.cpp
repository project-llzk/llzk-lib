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
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Parser/Parser.h>

#include <functional>
#include <gtest/gtest.h>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace llzk;

namespace {

/// Establish an ordered initial visit sequence while the real dataflow solver is running.
class IntervalStorageInitialization : public DataFlowAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IntervalStorageInitialization)

  IntervalStorageInitialization(DataFlowSolver &dataflowSolver, std::function<void()> callback)
      : DataFlowAnalysis(dataflowSolver), initializeCallback(std::move(callback)) {}

  LogicalResult initialize(Operation *) override {
    initializeCallback();
    return success(!testing::Test::HasFatalFailure());
  }
  LogicalResult visit(ProgramPoint *) override { return success(); }

private:
  std::function<void()> initializeCallback;
};

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

  using Visit = std::function<LogicalResult(Operation *)>;

  /// Initialize references, seed interval visits, then drain their real queued dependencies.
  void runOrdered(
      ModuleOp mod,
      const std::function<void(IntervalDataFlowAnalysis &, const Visit &, DataFlowSolver &)>
          &initialize,
      const std::function<void(IntervalDataFlowAnalysis &, DataFlowSolver &)> &check
  ) {
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    ASSERT_TRUE(succeeded(solver.initializeAndRun(mod)));
    auto *analysis = solver.load<IntervalDataFlowAnalysis>(smt, field, false, true);
    Visit visit = [&](Operation *op) {
      SmallVector<const IntervalAnalysisLattice *> operands;
      SmallVector<IntervalAnalysisLattice *> results;
      for (Value value : op->getOperands()) {
        auto *lattice = solver.getOrCreateState<IntervalAnalysisLattice>(value);
        lattice->useDefSubscribe(analysis);
        operands.push_back(lattice);
      }
      for (Value value : op->getResults()) {
        results.push_back(solver.getOrCreateState<IntervalAnalysisLattice>(value));
      }
      return analysis->visitOperation(op, operands, results);
    };
    auto *driver =
        solver.load<IntervalStorageInitialization>([&] { initialize(*analysis, visit, solver); });
    ASSERT_TRUE(succeeded(solver.initializeAndRun(mod, [&](DataFlowAnalysis &candidate) {
      return &candidate == driver;
    })));
    check(*analysis, solver);
  }

  /// Check a scalar result after the solver has propagated storage changes to its users.
  void expectValue(DataFlowSolver &solver, Value value, const UnreducedInterval &expected) {
    const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
    ASSERT_NE(lattice, nullptr);
    const auto &actual = lattice->getValue().getScalarValue();
    auto reduced = expected.reduce(field);
    EXPECT_TRUE(checkCond(reduced, actual.getInterval(), reduced == actual.getInterval()));
    EXPECT_EQ(actual.getOptionalUnreducedInterval(), expected);
  }

  /// Check both reduced and unreduced storage bounds without relying on aggregate SSA lattices.
  void
  expectStored(IntervalDataFlowAnalysis &analysis, const SourceRef &ref, const Interval &expected) {
    auto it = analysis.getWriteResults().find(ref);
    ASSERT_NE(it, analysis.getWriteResults().end()) << debug::toStringOne(ref);
    const auto &actual = it->second.getInterval();
    EXPECT_TRUE(checkCond(expected, actual, expected == actual));
    EXPECT_EQ(it->second.getOptionalUnreducedInterval(), expected.firstUnreduced());
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

// Independent valid indices select the same element only along singleton dimensions.
TEST_F(IntervalStorageTests, SingletonDimensionsPreserveReadAliases) {
  const std::tuple<StringRef, StringRef, StringRef, bool> cases[] = {
      {"1", "%i", "%j", true},           {"2", "%i", "%j", false},
      {"1,2", "%i, %j", "%k, %j", true}, {"1,2", "%i, %j", "%k, %l", false},
      {"2,1", "%i, %j", "%i, %k", true}, {"2,1", "%i, %j", "%k, %l", false},
      {"1,1", "%i, %j", "%k, %l", true}
  };
  for (auto [shape, leftIndices, rightIndices, sameElement] : cases) {
    SCOPED_TRACE(shape.str() + ": " + leftIndices.str() + " vs " + rightIndices.str());
    for (bool write : {false, true}) {
      SCOPED_TRACE(write ? "intervening write" : "no write");
      std::string source = R"mlir(
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<SHAPE x !felt.type>, %i: index, %j: index, %k: index, %l: index, %input: !felt.type) -> (!felt.type, !felt.type) {
    %seven = felt.const 7
    %nine = felt.const 9
    %left = array.read %array[LEFT_INDICES] : !array.type<SHAPE x !felt.type>, !felt.type
    // WRITE
    %right = array.read %array[RIGHT_INDICES] : !array.type<SHAPE x !felt.type>, !felt.type
    // CONSTRAINTS
    function.return %left, %right : !felt.type, !felt.type
  }
}
)mlir";
      if (write) {
        source.replace(
            source.find("// WRITE"), StringRef("// WRITE").size(),
            "array.write %array[LEFT_INDICES] = %input : !array.type<SHAPE x !felt.type>, "
            "!felt.type"
        );
      }
      for (auto [token, replacement] :
           {std::pair<StringRef, StringRef> {"SHAPE", shape},
            {"LEFT_INDICES", leftIndices},
            {"RIGHT_INDICES", rightIndices}}) {
        while (source.find(token.str()) != std::string::npos) {
          source.replace(source.find(token.str()), token.size(), replacement.str());
        }
      }
      for (bool constants : {false, true}) {
        checkReadConstraints(source, constants, sameElement && !write);
      }
    }
  }
}

// The singleton-index proof also applies to the prefix selected by an extraction.
TEST_F(IntervalStorageTests, SingletonSubarrayIndicesShareConstraints) {
  for (bool singleton : {false, true}) {
    SCOPED_TRACE(singleton ? "one row" : "independent rows");
    std::string source = R"mlir(
module attributes {llzk.lang} {
  function.def @test(%matrix: !array.type<ROWS,2 x !felt.type>, %i: index, %j: index, %k: index) -> (!felt.type, !felt.type) {
    %seven = felt.const 7
    %nine = felt.const 9
    %row1 = array.extract %matrix[%i] : !array.type<ROWS,2 x !felt.type>
    %row2 = array.extract %matrix[%k] : !array.type<ROWS,2 x !felt.type>
    %left = array.read %row1[%j] : !array.type<2 x !felt.type>, !felt.type
    %right = array.read %row2[%j] : !array.type<2 x !felt.type>, !felt.type
    // CONSTRAINTS
    function.return %left, %right : !felt.type, !felt.type
  }
}
)mlir";
    while (source.find("ROWS") != std::string::npos) {
      source.replace(source.find("ROWS"), StringRef("ROWS").size(), singleton ? "1" : "2");
    }
    for (bool constants : {false, true}) {
      checkReadConstraints(source, constants, singleton);
    }
  }
}

// Exact ranged writes lose their defaults only when concrete writes cover every candidate.
TEST_F(IntervalStorageTests, ExactRangedWritesRecomputeDefaultsFromConcreteCoverage) {
  for (bool rangedFirst : {false, true}) {
    SCOPED_TRACE(rangedFirst ? "ranged writes first" : "concrete writes first");
    auto mod = parseModule(R"mlir(
!Arrays = !pod.type<[@full: !array.type<2 x !felt.type>, @partial: !array.type<2 x !felt.type>, @optional: !array.type<2 x !felt.type>, @unbounded: !array.type<? x !felt.type>]>
!Values = !pod.type<[@full: !felt.type, @partial: !felt.type, @optional: !felt.type, @unbounded: !felt.type]>
module attributes {llzk.lang} {
  function.def @test(%full: !array.type<2 x !felt.type>, %partial: !array.type<2 x !felt.type>, %optional: !array.type<2 x !felt.type>, %unbounded: !array.type<? x !felt.type>, %i: index, %condition: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %seven = felt.const 7
    %nine = felt.const 9
    array.write %full[%i] = %nine : !array.type<2 x !felt.type>, !felt.type
    array.write %partial[%i] = %nine : !array.type<2 x !felt.type>, !felt.type
    array.write %optional[%i] = %nine : !array.type<2 x !felt.type>, !felt.type
    array.write %unbounded[%i] = %nine : !array.type<? x !felt.type>, !felt.type
    array.write %full[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
    array.write %full[%one] = %seven : !array.type<2 x !felt.type>, !felt.type
    array.write %partial[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
    array.write %optional[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
    array.write %unbounded[%zero] = %seven : !array.type<? x !felt.type>, !felt.type
    array.write %unbounded[%one] = %seven : !array.type<? x !felt.type>, !felt.type
    scf.if %condition {
      array.write %optional[%one] = %seven : !array.type<2 x !felt.type>, !felt.type
    }
    %fullRead = array.read %full[%i] : !array.type<2 x !felt.type>, !felt.type
    %partialRead = array.read %partial[%i] : !array.type<2 x !felt.type>, !felt.type
    %optionalRead = array.read %optional[%i] : !array.type<2 x !felt.type>, !felt.type
    %unboundedRead = array.read %unbounded[%i] : !array.type<? x !felt.type>, !felt.type
    %reads = pod.new { @full = %fullRead, @partial = %partialRead, @optional = %optionalRead, @unbounded = %unboundedRead } : !Values
    %copies = pod.new { @full = %full, @partial = %partial, @optional = %optional, @unbounded = %unbounded } : !Arrays
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto writes = llvm::to_vector(fn.getOps<array::WriteArrayOp>());
    auto conditional = *fn.getOps<scf::IfOp>().begin();
    auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        if (&op == writes.front().getOperation()) {
          break;
        }
        ASSERT_TRUE(succeeded(visit(&op)));
      }
      auto visitWrites = [&](bool ranged) {
        for (auto [i, write] : llvm::enumerate(writes)) {
          if ((i < 4) == ranged) {
            ASSERT_TRUE(succeeded(visit(write)));
          }
        }
        if (!ranged) {
          ASSERT_TRUE(succeeded(visit(&conditional.thenBlock()->front())));
        }
      };
      visitWrites(rangedFirst);
      visitWrites(!rangedFirst);
      // Read and copy only after all writes, independently of storage notifications.
      for (Operation *op = conditional->getNextNode(); op; op = op->getNextNode()) {
        ASSERT_TRUE(succeeded(visit(op)));
      }
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      llvm::StringRef names[] = {"full", "partial", "optional", "unbounded"};
      for (auto [i, name] : llvm::enumerate(names)) {
        SourceRefIndex record(StringAttr::get(&ctx, name));
        auto range = SourceRefIndex::forArrayDimension(i == 3 ? ShapedType::kDynamic : 2);
        auto expected = i == 0 ? UnreducedInterval(7, 9).reduce(field) : Interval::Entire(field);
        expectStored(analysis, SourceRef(pods[0]->getResult(0), {record}), expected);
        expectStored(analysis, SourceRef(pods[1]->getResult(0), {record, range}), expected);
      }
    });
  }
}

// Neither a ranged target nor one of several possible aliases is guaranteed to be written.
TEST_F(IntervalStorageTests, AmbiguousArrayWritesKeepNoWriteDefaults) {
  for (bool optionalFirst : {false, true}) {
    SCOPED_TRACE(optionalFirst ? "optional writes first" : "unconditional writes first");
    auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !felt.type>, %matrix: !array.type<2,2 x !felt.type>, %left: !array.type<2 x !felt.type>, %right: !array.type<2 x !felt.type>, %condition: i1, %chooseLeft: i1, %i: index, %j: index) {
    %zero = arith.constant 0 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %firstRow = array.extract %matrix[%i] : !array.type<2,2 x !felt.type>
    %secondRow = array.extract %matrix[%j] : !array.type<2,2 x !felt.type>
    %choice = scf.if %chooseLeft -> (!array.type<2 x !felt.type>) {
      scf.yield %left : !array.type<2 x !felt.type>
    } else {
      scf.yield %right : !array.type<2 x !felt.type>
    }
    scf.if %condition {
      array.write %array[%i] = %seven : !array.type<2 x !felt.type>, !felt.type
      array.write %firstRow[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
      array.write %left[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
      array.write %right[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
    }
    array.write %array[%j] = %nine : !array.type<2 x !felt.type>, !felt.type
    array.write %secondRow[%zero] = %nine : !array.type<2 x !felt.type>, !felt.type
    array.write %choice[%zero] = %nine : !array.type<2 x !felt.type>, !felt.type
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto conditional = llvm::to_vector(fn.getOps<scf::IfOp>()).back();
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        if (!isa<array::WriteArrayOp, scf::IfOp>(op)) {
          ASSERT_TRUE(succeeded(visit(&op)));
        }
      }
      auto visitWrites = [&](Block &block) {
        for (auto write : block.getOps<array::WriteArrayOp>()) {
          ASSERT_TRUE(succeeded(visit(write)));
        }
      };
      if (optionalFirst) {
        visitWrites(*conditional.thenBlock());
        visitWrites(fn.getBody().front());
      } else {
        visitWrites(fn.getBody().front());
        visitWrites(*conditional.thenBlock());
      }
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      auto range = SourceRefIndex::forArrayDimension(2);
      expectStored(analysis, SourceRef(fn.getArgument(0), {range}), Interval::Entire(field));
      expectStored(
          analysis, SourceRef(fn.getArgument(1), {range, SourceRefIndex(0)}),
          Interval::Entire(field)
      );
      for (unsigned arg : {2U, 3U}) {
        expectStored(
            analysis, SourceRef(fn.getArgument(arg), {SourceRefIndex(0)}), Interval::Entire(field)
        );
      }
    });
  }
}

// A write through a merged POD can miss either candidate, including aggregate-valued fields.
TEST_F(IntervalStorageTests, AmbiguousPodWritesKeepNoWriteDefaults) {
  for (bool optionalFirst : {false, true}) {
    SCOPED_TRACE(optionalFirst ? "optional writes first" : "unconditional writes first");
    auto mod = parseModule(R"mlir(
!Inner = !pod.type<[@value: !felt.type]>
!P = !pod.type<[@value: !felt.type, @nested: !Inner]>
module attributes {llzk.lang} {
  function.def @test(%left: !P, %right: !P, %condition: i1, %chooseLeft: i1) {
    %seven = felt.const 7
    %nine = felt.const 9
    %sevenPod = pod.new { @value = %seven } : !Inner
    %ninePod = pod.new { @value = %nine } : !Inner
    %choice = scf.if %chooseLeft -> (!P) {
      scf.yield %left : !P
    } else {
      scf.yield %right : !P
    }
    scf.if %condition {
      pod.write %left[@value] = %seven : !P, !felt.type
      pod.write %right[@value] = %seven : !P, !felt.type
      pod.write %left[@nested] = %sevenPod : !P, !Inner
      pod.write %right[@nested] = %sevenPod : !P, !Inner
    }
    pod.write %choice[@value] = %nine : !P, !felt.type
    pod.write %choice[@nested] = %ninePod : !P, !Inner
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto conditional = llvm::to_vector(fn.getOps<scf::IfOp>()).back();
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        if (!isa<pod::WritePodOp, scf::IfOp>(op)) {
          ASSERT_TRUE(succeeded(visit(&op)));
        }
      }
      auto visitWrites = [&](Block &block) {
        for (auto write : block.getOps<pod::WritePodOp>()) {
          ASSERT_TRUE(succeeded(visit(write)));
        }
      };
      if (optionalFirst) {
        visitWrites(*conditional.thenBlock());
        visitWrites(fn.getBody().front());
      } else {
        visitWrites(fn.getBody().front());
        visitWrites(*conditional.thenBlock());
      }
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      SourceRefIndex value(StringAttr::get(&ctx, "value"));
      SourceRefIndex nested(StringAttr::get(&ctx, "nested"));
      for (unsigned arg : {0U, 1U}) {
        expectStored(analysis, SourceRef(fn.getArgument(arg), {value}), Interval::Entire(field));
        expectStored(
            analysis, SourceRef(fn.getArgument(arg), {nested, value}), Interval::Entire(field)
        );
      }
    });
  }
}

// An unconditional write must remove the no-write alternative in either visitation order.
TEST_F(IntervalStorageTests, UnconditionalWritesRemoveNoWriteDefaults) {
  for (bool optionalFirst : {false, true}) {
    SCOPED_TRACE(optionalFirst ? "optional writes first" : "unconditional writes first");
    auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type, @other: !felt.type]>
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %pod = pod.new : !P
    %array = array.new : !array.type<2 x !felt.type>
    scf.if %condition {
      pod.write %pod[@value] = %seven : !P, !felt.type
      pod.write %pod[@other] = %seven : !P, !felt.type
      array.write %array[%zero] = %seven : !array.type<2 x !felt.type>, !felt.type
      array.write %array[%one] = %seven : !array.type<2 x !felt.type>, !felt.type
    }
    pod.write %pod[@value] = %nine : !P, !felt.type
    array.write %array[%zero] = %nine : !array.type<2 x !felt.type>, !felt.type
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto podValue = *fn.getOps<pod::NewPodOp>().begin();
    auto arrayValue = *fn.getOps<array::CreateArrayOp>().begin();
    auto conditional = *fn.getOps<scf::IfOp>().begin();
    auto isWrite = [](Operation *op) { return isa<pod::WritePodOp, array::WriteArrayOp>(op); };
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        if (!isWrite(&op) && !isa<scf::IfOp>(op)) {
          ASSERT_TRUE(succeeded(visit(&op)));
        }
      }
      auto visitWrites = [&](Block &block) {
        for (Operation &op : block) {
          if (isWrite(&op)) {
            ASSERT_TRUE(succeeded(visit(&op)));
          }
        }
      };
      if (optionalFirst) {
        visitWrites(*conditional.thenBlock());
        visitWrites(fn.getBody().front());
      } else {
        visitWrites(fn.getBody().front());
        visitWrites(*conditional.thenBlock());
      }
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      auto joined = UnreducedInterval(7, 9).reduce(field);
      expectStored(
          analysis,
          SourceRef(podValue->getResult(0), {SourceRefIndex(StringAttr::get(&ctx, "value"))}),
          joined
      );
      expectStored(analysis, SourceRef(arrayValue->getResult(0), {SourceRefIndex(0)}), joined);
      // A guaranteed write to one field does not eliminate defaults at other locations.
      expectStored(
          analysis,
          SourceRef(podValue->getResult(0), {SourceRefIndex(StringAttr::get(&ctx, "other"))}),
          Interval::Entire(field)
      );
      expectStored(
          analysis, SourceRef(arrayValue->getResult(0), {SourceRefIndex(1)}),
          Interval::Entire(field)
      );
    });
  }
}

// A later contribution must re-expand existing reads and every dependent calculation.
TEST_F(IntervalStorageTests, GrowingWritesReexpandReadersAndUsers) {
  for (bool optional : {false, true}) {
    SCOPED_TRACE(optional ? "optional third write" : "unconditional third write");
    for (bool late : {false, true}) {
      SCOPED_TRACE(late ? "third write after readers" : "third write before readers");
      auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
module attributes {llzk.lang} {
  function.def @test(%pod: !P, %array: !array.type<1 x !felt.type>, %condition: i1) {
    %zero = arith.constant 0 : index
    %one = felt.const 1
    %seven = felt.const 7
    %nine = felt.const 9
    %eleven = felt.const 11
    scf.if %condition {
      pod.write %pod[@value] = %seven : !P, !felt.type
      array.write %array[%zero] = %seven : !array.type<1 x !felt.type>, !felt.type
    }
    pod.write %pod[@value] = %nine : !P, !felt.type
    array.write %array[%zero] = %nine : !array.type<1 x !felt.type>, !felt.type
    scf.if %condition {
      pod.write %pod[@value] = %eleven : !P, !felt.type
      array.write %array[%zero] = %eleven : !array.type<1 x !felt.type>, !felt.type
    }
    %podRead = pod.read %pod[@value] : !P, !felt.type
    %arrayRead = array.read %array[%zero] : !array.type<1 x !felt.type>, !felt.type
    %podSum = felt.add %podRead, %one
    %arraySum = felt.add %arrayRead, %one
    %podNeg = felt.neg %podSum
    %arrayNeg = felt.neg %arraySum
    %podSelect = arith.select %condition, %podSum, %one : !felt.type
    %arraySelect = arith.select %condition, %arraySum, %one : !felt.type
    %podNext = felt.add %podSelect, %one
    %arrayNext = felt.add %arraySelect, %one
    function.return
  }
}
)mlir");
      ASSERT_TRUE(mod);
      auto fn = *mod->getOps<function::FuncDefOp>().begin();
      auto branches = llvm::to_vector(fn.getOps<scf::IfOp>());
      auto isWrite = [](Operation *op) { return isa<pod::WritePodOp, array::WriteArrayOp>(op); };
      SmallVector<Operation *> firstWrites, guaranteedWrites, thirdWrites, calculations;
      for (Operation &op : *branches.front().thenBlock()) {
        if (isWrite(&op)) {
          firstWrites.push_back(&op);
        }
      }
      for (Operation &op : fn.getBody().front()) {
        if (isWrite(&op)) {
          guaranteedWrites.push_back(&op);
        } else if (
            isa<pod::ReadPodOp, array::ReadArrayOp, felt::AddFeltOp, felt::NegFeltOp,
                arith::SelectOp>(op)
        ) {
          calculations.push_back(&op);
        }
      }
      for (Operation &op : *branches.back().thenBlock()) {
        if (isWrite(&op)) {
          thirdWrites.push_back(&op);
        }
      }
      if (!optional) {
        for (Operation *op : thirdWrites) {
          op->moveBefore(branches.back());
        }
      }
      auto checkValues = [&](DataFlowSolver &solver, int high) {
        SmallVector<UnreducedInterval> expected {
            {7, high}, {8, high + 1}, {-high - 1, -8}, {1, high + 1}, {2, high + 2}
        };
        ASSERT_EQ(calculations.size(), 2 * expected.size());
        for (auto [i, op] : llvm::enumerate(calculations)) {
          SCOPED_TRACE(debug::toStringOne(*op));
          expectValue(solver, op->getResult(0), expected[i / 2]);
        }
      };
      runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &solver) {
        for (Operation &op : fn.getBody().front()) {
          if (isa<arith::ConstantOp, felt::FeltConstantOp>(op)) {
            ASSERT_TRUE(succeeded(visit(&op)));
          }
        }
        auto visitAll = [&](ArrayRef<Operation *> ops) {
          for (Operation *op : ops) {
            ASSERT_TRUE(succeeded(visit(op)));
          }
        };
        visitAll(firstWrites);
        visitAll(guaranteedWrites);
        if (!late) {
          visitAll(thirdWrites);
        }
        visitAll(calculations);
        checkValues(solver, late ? 9 : 11);
        if (late) {
          visitAll(thirdWrites);
        }
      }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &solver) {
        auto joined = UnreducedInterval(7, 11).reduce(field);
        expectStored(
            analysis,
            SourceRef(fn.getArgument(0), {SourceRefIndex(StringAttr::get(&ctx, "value"))}), joined
        );
        expectStored(analysis, SourceRef(fn.getArgument(1), {SourceRefIndex(0)}), joined);
        checkValues(solver, 11);
      });
    }
  }
}

// Backward bounds inferred from an earlier read must be recomputed after expansion too.
TEST_F(IntervalStorageTests, GrowingWritesRecomputeConstraintRefinements) {
  for (unsigned order : {0U, 1U, 2U}) {
    // Visit the write before constraints, after constraints, or after a lazily resolved read.
    SCOPED_TRACE(order);
    bool late = order != 0;
    auto mod = parseModule(R"mlir(
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<1 x !felt.type>, %condition: i1, %input: !felt.type)
      attributes {function.allow_constraint} {
    %zero = arith.constant 0 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %eleven = felt.const 11
    %twenty = felt.const 20
    scf.if %condition {
      array.write %array[%zero] = %seven : !array.type<1 x !felt.type>, !felt.type
    }
    array.write %array[%zero] = %nine : !array.type<1 x !felt.type>, !felt.type
    scf.if %condition {
      array.write %array[%zero] = %eleven : !array.type<1 x !felt.type>, !felt.type
    }
    %read = array.read %array[%zero] : !array.type<1 x !felt.type>, !felt.type
    %sum = felt.add %read, %input
    constrain.eq %sum, %twenty : !felt.type
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto branches = llvm::to_vector(fn.getOps<scf::IfOp>());
    auto first = *branches.front().thenBlock()->getOps<array::WriteArrayOp>().begin();
    auto third = *branches.back().thenBlock()->getOps<array::WriteArrayOp>().begin();
    auto guaranteed = *fn.getOps<array::WriteArrayOp>().begin();
    auto read = *fn.getOps<array::ReadArrayOp>().begin();
    auto sum = *fn.getOps<felt::AddFeltOp>().begin();
    auto constraint = *fn.getOps<constrain::EmitEqualityOp>().begin();
    auto checkReduced = [&](DataFlowSolver &solver, Value value, int low, int high) {
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
      ASSERT_NE(lattice, nullptr);
      const auto &actual = lattice->getValue().getScalarValue().getInterval();
      auto expected = UnreducedInterval(low, high).reduce(field);
      EXPECT_TRUE(checkCond(expected, actual, expected == actual));
    };
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &solver) {
      for (Operation &op : fn.getBody().front()) {
        if (isa<arith::ConstantOp, felt::FeltConstantOp>(op)) {
          ASSERT_TRUE(succeeded(visit(&op)));
        }
      }
      ASSERT_TRUE(succeeded(visit(first)));
      ASSERT_TRUE(succeeded(visit(guaranteed)));
      if (!late) {
        ASSERT_TRUE(succeeded(visit(third)));
      }
      if (order != 2) {
        ASSERT_TRUE(succeeded(visit(read)));
      }
      ASSERT_TRUE(succeeded(visit(sum)));
      ASSERT_TRUE(succeeded(visit(constraint)));
      checkReduced(solver, fn.getArgument(2), late ? 11 : 9, 13);
      if (late) {
        ASSERT_TRUE(succeeded(visit(third)));
      }
    }, [&](IntervalDataFlowAnalysis &, DataFlowSolver &solver) {
      checkReduced(solver, read.getResult(), 7, 11);
      checkReduced(solver, sum.getResult(), 20, 20);
      checkReduced(solver, fn.getArgument(2), 9, 13);
    });
  }
}

// Analyze every write first to isolate overlap matching from dependency notifications.
TEST_F(IntervalStorageTests, AggregateCopiesIncludeOverlappingWrites) {
  for (bool rangedSource : {false, true}) {
    SCOPED_TRACE(rangedSource ? "ranged source" : "concrete source");
    auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type, @other: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  struct.def @Box {
    struct.member @stored : !P
    function.def @compute(%array: !array.type<2 x !P>, %i: index, %j: index) -> !struct.type<@Box> {
      %box = struct.new : !struct.type<@Box>
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %seven = felt.const 7
      %nine = felt.const 9
      %first = array.read %array[%zero] : !array.type<2 x !P>, !P
      %second = array.read %array[%one] : !array.type<2 x !P>, !P
      %dynamic = array.read %array[%i] : !array.type<2 x !P>, !P
      pod.write %first[@value] = %seven : !P, !felt.type
      pod.write %second[@value] = %seven : !P, !felt.type
      pod.write %first[@other] = %seven : !P, !felt.type
      pod.write %second[@other] = %seven : !P, !felt.type
      pod.write %dynamic[@value] = %nine : !P, !felt.type
      %source = array.read %array[%j] : !array.type<2 x !P>, !P
      %podCopy = pod.new { @child = %source } : !W
      %arrayCopy = array.new %source : !array.type<1 x !P>
      %assigned = pod.new : !W
      pod.write %assigned[@child] = %source : !W, !P
      struct.writem %box[@stored] = %source : !struct.type<@Box>, !P
      function.return %box : !struct.type<@Box>
    }
    function.def @constrain(%self: !struct.type<@Box>, %array: !array.type<2 x !P>, %i: index, %j: index) {
      function.return
    }
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto box = *mod->getOps<component::StructDefOp>().begin();
    auto fn = *box.getOps<function::FuncDefOp>().begin();
    auto reads = llvm::to_vector(fn.getOps<array::ReadArrayOp>());
    if (!rangedSource) {
      reads.back()->setOperand(1, reads[1].getIndices().front());
    }
    auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
    auto arrayCopy = *fn.getOps<array::CreateArrayOp>().begin();
    auto member = *box.getOps<component::MemberDefOp>().begin();
    SourceRefIndex child(StringAttr::get(&ctx, "child"));
    SourceRefIndex value(StringAttr::get(&ctx, "value"));
    SourceRefIndex other(StringAttr::get(&ctx, "other"));
    SmallVector<SourceRef> copies {
        SourceRef(pods[0]->getResult(0), {child}),
        SourceRef(arrayCopy->getResult(0), {SourceRefIndex(0)}),
        SourceRef(pods[1]->getResult(0), {child}),
        SourceRef(
            (*fn.getOps<component::CreateStructOp>().begin())->getResult(0),
            {SourceRefIndex(member)}
        )
    };
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        ASSERT_TRUE(succeeded(visit(&op)));
      }
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      for (const SourceRef &copy : copies) {
        expectStored(analysis, *copy.createChild(value), UnreducedInterval(7, 9).reduce(field));
        expectStored(
            analysis, *copy.createChild(other), Interval::Degenerate(field, field.felt(7))
        );
      }
    });
  }
}

// Copying two independent dynamic selections must preserve their separate value identities.
TEST_F(IntervalStorageTests, RangedAggregateCopiesKeepIndependentSymbols) {
  auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !P>, %i: index, %j: index, %input: !felt.type) {
    %left = array.read %array[%i] : !array.type<2 x !P>, !P
    %right = array.read %array[%j] : !array.type<2 x !P>, !P
    pod.write %left[@value] = %input : !P, !felt.type
    %leftCopy = pod.new { @child = %left } : !W
    %rightCopy = pod.new { @child = %right } : !W
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  auto copies = llvm::to_vector(fn.getOps<pod::NewPodOp>());
  SourceRef::Path path {
      SourceRefIndex(StringAttr::get(&ctx, "child")), SourceRefIndex(StringAttr::get(&ctx, "value"))
  };
  runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
    for (Operation &op : fn.getBody().front()) {
      ASSERT_TRUE(succeeded(visit(&op)));
    }
  }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
    SmallVector<llvm::SMTExprRef> expressions;
    for (auto copy : copies) {
      SourceRef ref(copy->getResult(0), path);
      expectStored(analysis, ref, Interval::Entire(field));
      auto it = analysis.getWriteResults().find(ref);
      ASSERT_NE(it, analysis.getWriteResults().end());
      expressions.push_back(it->second.getExpr());
      ASSERT_NE(expressions.back(), nullptr);
    }
    EXPECT_FALSE(*expressions[0] == *expressions[1]);
  });
}

// A partially written or unbounded selection still includes unwritten alternatives.
TEST_F(IntervalStorageTests, RangedAggregateCopiesPreserveUnwrittenDefaults) {
  auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !P>, %dynamic: !array.type<? x !P>, %i: index) {
    %zero = arith.constant 0 : index
    %seven = felt.const 7
    %first = array.read %array[%zero] : !array.type<2 x !P>, !P
    %dynamicFirst = array.read %dynamic[%zero] : !array.type<? x !P>, !P
    pod.write %first[@value] = %seven : !P, !felt.type
    pod.write %dynamicFirst[@value] = %seven : !P, !felt.type
    %partial = array.read %array[%i] : !array.type<2 x !P>, !P
    %unbounded = array.read %dynamic[%i] : !array.type<? x !P>, !P
    %partialCopy = pod.new { @child = %partial } : !W
    %unboundedCopy = pod.new { @child = %unbounded } : !W
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
    for (Operation &op : fn.getBody().front()) {
      ASSERT_TRUE(succeeded(visit(&op)));
    }
  }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
    for (auto copy : fn.getOps<pod::NewPodOp>()) {
      expectStored(
          analysis,
          SourceRef(
              copy->getResult(0), {SourceRefIndex(StringAttr::get(&ctx, "child")),
                                   SourceRefIndex(StringAttr::get(&ctx, "value"))}
          ),
          Interval::Entire(field)
      );
    }
  });
}

// Storage notifications must cross every copy, including materialized array roots and reads.
TEST_F(IntervalStorageTests, DelayedWritesReachEveryAggregateCopy) {
  for (bool delayedWrite : {false, true}) {
    SCOPED_TRACE(delayedWrite);
    auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P]>
!V = !pod.type<[@values: !array.type<2 x !W>]>
module attributes {llzk.lang} {
  function.def @test() {
    %index = arith.constant 1 : index
    %seven = felt.const 7
    %source = pod.new : !P
    pod.write %source[@value] = %seven : !P, !felt.type
    %copy1 = pod.new { @child = %source } : !W
    %alias = pod.read %copy1[@child] : !W, !P
    %copy2 = pod.new { @child = %alias } : !W
    %array = array.new %copy2, %copy2 : !array.type<2 x !W>
    %copy3 = pod.new { @values = %array } : !V
    %values = pod.read %copy3[@values] : !V, !array.type<2 x !W>
    %element = array.read %values[%index] : !array.type<2 x !W>, !W
    %child = pod.read %element[@child] : !W, !P
    %value = pod.read %child[@value] : !P, !felt.type
    %scalarCopy = pod.new { @value = %value } : !P
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto write = *fn.getOps<pod::WritePodOp>().begin();
    auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
    auto array = *fn.getOps<array::CreateArrayOp>().begin();
    SourceRefIndex child(StringAttr::get(&ctx, "child"));
    SourceRefIndex value(StringAttr::get(&ctx, "value"));
    SourceRefIndex values(StringAttr::get(&ctx, "values"));
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        if (delayedWrite && &op == write.getOperation()) {
          continue;
        }
        ASSERT_TRUE(succeeded(visit(&op)));
      }
      if (delayedWrite) {
        ASSERT_TRUE(succeeded(visit(write)));
      }
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      auto expected = Interval::Degenerate(field, field.felt(7));
      expectStored(analysis, SourceRef(pods[0]->getResult(0), {value}), expected);
      for (unsigned i : {1U, 2U}) {
        expectStored(analysis, SourceRef(pods[i]->getResult(0), {child, value}), expected);
      }
      for (int64_t index : {0, 1}) {
        expectStored(
            analysis, SourceRef(array->getResult(0), {SourceRefIndex(index), child, value}),
            expected
        );
        expectStored(
            analysis,
            SourceRef(pods[3]->getResult(0), {values, SourceRefIndex(index), child, value}),
            expected
        );
      }
      expectStored(analysis, SourceRef(pods[4]->getResult(0), {value}), expected);
    });
  }
}

// Replace a copy's stale default without discarding independent writes to its destination.
TEST_F(IntervalStorageTests, RecomputedCopiesReplaceDefaultsAndPreserveOtherWrites) {
  auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %seven = felt.const 7
    %nine = felt.const 9
    %source = pod.new : !P
    scf.if %condition {
      pod.write %source[@value] = %seven : !P, !felt.type
    }
    pod.write %source[@value] = %seven : !P, !felt.type
    %copy = pod.new { @child = %source } : !W
    %alias = pod.read %copy[@child] : !W, !P
    scf.if %condition {
      pod.write %alias[@value] = %nine : !P, !felt.type
    }
    %out = pod.new { @child = %alias } : !W
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  auto guaranteed = *fn.getOps<pod::WritePodOp>().begin();
  auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
  SourceRefIndex child(StringAttr::get(&ctx, "child"));
  SourceRefIndex value(StringAttr::get(&ctx, "value"));
  runOrdered(*mod, [&](IntervalDataFlowAnalysis &analysis, const Visit &visit, DataFlowSolver &) {
    for (Operation &op : fn.getBody().front()) {
      if (&op == guaranteed.getOperation()) {
        continue;
      }
      if (auto conditional = dyn_cast<scf::IfOp>(op)) {
        ASSERT_TRUE(succeeded(visit(&conditional.thenBlock()->front())));
      } else {
        ASSERT_TRUE(succeeded(visit(&op)));
      }
    }
    expectStored(
        analysis, SourceRef(pods.back()->getResult(0), {child, value}), Interval::Entire(field)
    );
    ASSERT_TRUE(succeeded(visit(guaranteed)));
  }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
    expectStored(
        analysis, SourceRef(pods.front()->getResult(0), {value}),
        Interval::Degenerate(field, field.felt(7))
    );
    for (unsigned i : {1U, 2U}) {
      expectStored(
          analysis, SourceRef(pods[i]->getResult(0), {child, value}),
          UnreducedInterval(7, 9).reduce(field)
      );
    }
  });
}

// A revisited yield replaces its stale default while preserving the other branch's contribution.
TEST_F(IntervalStorageTests, RecomputedYieldsReplaceDefaultsAndPreserveOtherBranches) {
  for (bool separateSource : {false, true}) {
    SCOPED_TRACE(separateSource ? "distinct branch sources" : "same branch source");
    std::string source = R"mlir(
!P = !pod.type<[@value: !felt.type]>
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %seven = felt.const 7
    %nine = felt.const 9
    %source = pod.new : !P
    scf.if %condition {
      pod.write %source[@value] = %seven : !P, !felt.type
    }
    pod.write %source[@value] = %seven : !P, !felt.type
    %other = pod.new { @value = %nine } : !P
    %result = scf.if %condition -> (!P) {
      scf.yield %source : !P
    } else {
      scf.yield ELSE_SOURCE : !P
    }
    function.return
  }
}
)mlir";
    source.replace(
        source.find("ELSE_SOURCE"), std::string("ELSE_SOURCE").size(),
        separateSource ? "%other" : "%source"
    );
    auto mod = parseModule(source);
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto guaranteed = *fn.getOps<pod::WritePodOp>().begin();
    auto conditionals = llvm::to_vector(fn.getOps<scf::IfOp>());
    auto sourcePod = *fn.getOps<pod::NewPodOp>().begin();
    SourceRefIndex value(StringAttr::get(&ctx, "value"));
    SourceRef resultRef(conditionals.back()->getResult(0), {value});
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &analysis, const Visit &visit) {
      for (Operation &op : fn.getBody().front()) {
        if (&op == guaranteed.getOperation()) {
          continue;
        }
        if (&op == conditionals.front().getOperation()) {
          ASSERT_TRUE(succeeded(visit(&conditionals.front().thenBlock()->front())));
        } else if (&op == conditionals.back().getOperation()) {
          ASSERT_TRUE(succeeded(visit(conditionals.back().thenBlock()->getTerminator())));
          ASSERT_TRUE(succeeded(visit(conditionals.back().elseBlock()->getTerminator())));
        } else {
          ASSERT_TRUE(succeeded(visit(&op)));
        }
      }
      expectStored(analysis, resultRef, Interval::Entire(field));
      ASSERT_TRUE(succeeded(visit(guaranteed)));
    }, [&](IntervalDataFlowAnalysis &analysis) {
      expectStored(
          analysis, SourceRef(sourcePod->getResult(0), {value}),
          Interval::Degenerate(field, field.felt(7))
      );
      expectStored(
          analysis, resultRef,
          separateSource ? UnreducedInterval(7, 9).reduce(field)
                         : Interval::Degenerate(field, field.felt(7))
      );
    });
  }
}

// One yield can forward several possible sources; all must contribute to each result leaf.
TEST_F(IntervalStorageTests, YieldRemapsPreserveAliasedSources) {
  auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!A = !array.type<1 x !felt.type>
module attributes {llzk.lang} {
  function.def @test(%condition: i1) {
    %zero = arith.constant 0 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %left = pod.new { @value = %seven } : !P
    %right = pod.new { @value = %nine } : !P
    %leftArray = array.new : !A
    %rightArray = array.new : !A
    array.write %leftArray[%zero] = %seven : !A, !felt.type
    array.write %rightArray[%zero] = %nine : !A, !felt.type
    %selected, %selectedArray = scf.if %condition -> (!P, !A) {
      scf.yield %left, %leftArray : !P, !A
    } else {
      scf.yield %right, %rightArray : !P, !A
    }
    %result, %resultArray = scf.execute_region -> (!P, !A) {
      scf.yield %selected, %selectedArray : !P, !A
    }
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  auto result = *fn.getOps<scf::ExecuteRegionOp>().begin();
  runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit) {
    for (Operation &op : fn.getBody().front()) {
      if (isa<scf::IfOp, scf::ExecuteRegionOp>(op)) {
        for (Region &region : op.getRegions()) {
          ASSERT_TRUE(succeeded(visit(region.front().getTerminator())));
        }
      } else {
        ASSERT_TRUE(succeeded(visit(&op)));
      }
    }
  }, [&](IntervalDataFlowAnalysis &analysis) {
    auto expected = UnreducedInterval(7, 9).reduce(field);
    SourceRefIndex value(StringAttr::get(&ctx, "value"));
    expectStored(analysis, SourceRef(result->getResult(0), {value}), expected);
    expectStored(analysis, SourceRef(result->getResult(1), {SourceRefIndex(0)}), expected);
  });
}

// Notify overlapping prefixes in both directions; disjoint writes must not alter copies.
TEST_F(IntervalStorageTests, AggregateSubscribersObserveOverlappingWrites) {
  for (unsigned selection : {0U, 1U, 2U}) {
    SCOPED_TRACE(selection); // Ranged reader, ranged writer, disjoint concrete selections.
    auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type, @other: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  function.def @test(%array: !array.type<2 x !P>, %i: index) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %first = array.read %array[%zero] : !array.type<2 x !P>, !P
    %second = array.read %array[%one] : !array.type<2 x !P>, !P
    %dynamic = array.read %array[%i] : !array.type<2 x !P>, !P
    pod.write %first[@value] = %seven : !P, !felt.type
    pod.write %second[@value] = %seven : !P, !felt.type
    pod.write %first[@other] = %seven : !P, !felt.type
    pod.write %second[@other] = %seven : !P, !felt.type
    pod.write %second[@value] = %nine : !P, !felt.type
    %copy = pod.new { @child = %dynamic } : !W
    %alias = pod.read %copy[@child] : !W, !P
    %out = pod.new { @child = %alias } : !W
    function.return
  }
}
)mlir");
    ASSERT_TRUE(mod);
    auto fn = *mod->getOps<function::FuncDefOp>().begin();
    auto reads = llvm::to_vector(fn.getOps<array::ReadArrayOp>());
    auto lateWrite = llvm::to_vector(fn.getOps<pod::WritePodOp>()).back();
    auto copies = llvm::to_vector(fn.getOps<pod::NewPodOp>());
    if (selection != 0) {
      copies.front()->setOperand(0, reads.front().getResult());
    }
    if (selection == 1) {
      lateWrite->setOperand(0, reads.back().getResult());
    }
    SourceRefIndex child(StringAttr::get(&ctx, "child"));
    SourceRefIndex value(StringAttr::get(&ctx, "value"));
    SourceRefIndex other(StringAttr::get(&ctx, "other"));
    runOrdered(*mod, [&](IntervalDataFlowAnalysis &analysis, const Visit &visit, DataFlowSolver &) {
      for (Operation &op : fn.getBody().front()) {
        if (&op != lateWrite.getOperation()) {
          ASSERT_TRUE(succeeded(visit(&op)));
        }
      }
      for (auto copy : copies) {
        expectStored(
            analysis, SourceRef(copy->getResult(0), {child, value}),
            Interval::Degenerate(field, field.felt(7))
        );
      }
      ASSERT_TRUE(succeeded(visit(lateWrite)));
    }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
      auto expected = selection == 2 ? Interval::Degenerate(field, field.felt(7))
                                     : UnreducedInterval(7, 9).reduce(field);
      for (auto copy : copies) {
        expectStored(analysis, SourceRef(copy->getResult(0), {child, value}), expected);
        expectStored(
            analysis, SourceRef(copy->getResult(0), {child, other}),
            Interval::Degenerate(field, field.felt(7))
        );
      }
    });
  }
}

// Scalar initializer publications must refine when their SSA producer is analyzed later.
TEST_F(IntervalStorageTests, ScalarInitializersReplaceRecomputedValues) {
  auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P, @values: !array.type<1 x !felt.type>]>
module attributes {llzk.lang} {
  function.def @test() {
    %three = felt.const 3
    %four = felt.const 4
    %sum = felt.add %three, %four
    %pod = pod.new { @value = %sum } : !P
    %array = array.new %sum : !array.type<1 x !felt.type>
    %out = pod.new { @child = %pod, @values = %array } : !W
    function.return
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto fn = *mod->getOps<function::FuncDefOp>().begin();
  auto sum = *fn.getOps<felt::AddFeltOp>().begin();
  auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
  auto array = *fn.getOps<array::CreateArrayOp>().begin();
  SourceRefIndex child(StringAttr::get(&ctx, "child"));
  SourceRefIndex value(StringAttr::get(&ctx, "value"));
  SourceRefIndex values(StringAttr::get(&ctx, "values"));
  runOrdered(*mod, [&](IntervalDataFlowAnalysis &analysis, const Visit &visit, DataFlowSolver &) {
    for (Operation &op : fn.getBody().front()) {
      if (&op != sum.getOperation()) {
        ASSERT_TRUE(succeeded(visit(&op)));
      }
    }
    expectStored(analysis, SourceRef(pods.front()->getResult(0), {value}), Interval::Entire(field));
    ASSERT_TRUE(succeeded(visit(sum)));
  }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
    auto expected = Interval::Degenerate(field, field.felt(7));
    expectStored(analysis, SourceRef(pods.front()->getResult(0), {value}), expected);
    expectStored(analysis, SourceRef(array->getResult(0), {SourceRefIndex(0)}), expected);
    expectStored(analysis, SourceRef(pods.back()->getResult(0), {child, value}), expected);
    expectStored(
        analysis, SourceRef(pods.back()->getResult(0), {values, SourceRefIndex(0)}), expected
    );
  });
}

// Delayed fields also cross aggregate assignments and component member storage.
TEST_F(IntervalStorageTests, DelayedFieldsReachAggregateAssignments) {
  auto mod = parseModule(R"mlir(
!P = !pod.type<[@value: !felt.type]>
!W = !pod.type<[@child: !P]>
module attributes {llzk.lang} {
  struct.def @Box {
    struct.member @stored : !W
    function.def @compute() -> !struct.type<@Box> {
      %self = struct.new : !struct.type<@Box>
      %seven = felt.const 7
      %source = pod.new : !P
      pod.write %source[@value] = %seven : !P, !felt.type
      %assigned = pod.new : !W
      pod.write %assigned[@child] = %source : !W, !P
      struct.writem %self[@stored] = %assigned : !struct.type<@Box>, !W
      %member = struct.readm %self[@stored] : !struct.type<@Box>, !W
      %child = pod.read %member[@child] : !W, !P
      %copy = pod.new { @child = %child } : !W
      function.return %self : !struct.type<@Box>
    }
    function.def @constrain(%self: !struct.type<@Box>) {
      function.return
    }
  }
}
)mlir");
  ASSERT_TRUE(mod);
  auto box = *mod->getOps<component::StructDefOp>().begin();
  auto member = *box.getOps<component::MemberDefOp>().begin();
  auto fn = *box.getOps<function::FuncDefOp>().begin();
  auto self = *fn.getOps<component::CreateStructOp>().begin();
  auto write = *fn.getOps<pod::WritePodOp>().begin();
  auto pods = llvm::to_vector(fn.getOps<pod::NewPodOp>());
  SourceRefIndex child(StringAttr::get(&ctx, "child"));
  SourceRefIndex value(StringAttr::get(&ctx, "value"));
  runOrdered(*mod, [&](IntervalDataFlowAnalysis &, const Visit &visit, DataFlowSolver &) {
    for (Operation &op : fn.getBody().front()) {
      if (&op != write.getOperation()) {
        ASSERT_TRUE(succeeded(visit(&op)));
      }
    }
    ASSERT_TRUE(succeeded(visit(write)));
  }, [&](IntervalDataFlowAnalysis &analysis, DataFlowSolver &) {
    auto expected = Interval::Degenerate(field, field.felt(7));
    for (unsigned i : {1U, 2U}) {
      expectStored(analysis, SourceRef(pods[i]->getResult(0), {child, value}), expected);
    }
    expectStored(
        analysis, SourceRef(self->getResult(0), {SourceRefIndex(member), child, value}), expected
    );
  });
}

// Materializing array value storage must not replace known component handle identities.
TEST_F(IntervalStorageTests, ArrayCopiesPreserveComponentConstraintIdentity) {
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
  function.def @test(%a: !struct.type<@Cell>, %b: !struct.type<@Cell>) -> (!felt.type, !felt.type)
      attributes {function.allow_constraint} {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %seven = felt.const 7
    %nine = felt.const 9
    %array = array.new %a, %b : !array.type<2 x !struct.type<@Cell>>
    %left = array.read %array[%zero] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %right = array.read %array[%one] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %x = struct.readm %left[@value] : !struct.type<@Cell>, !felt.type
    %y = struct.readm %right[@value] : !struct.type<@Cell>, !felt.type
    %leftCheck = bool.cmp eq(%x, %seven)
    bool.assert %leftCheck
    %rightCheck = bool.cmp eq(%y, %nine)
    bool.assert %rightCheck
    function.return %x, %y : !felt.type, !felt.type
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
  auto cell = *mod->getOps<component::StructDefOp>().begin();
  auto member = *cell.getOps<component::MemberDefOp>().begin();
  for (auto [i, value] : llvm::enumerate(fn.getBody().front().getTerminator()->getOperands())) {
    const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(value);
    ASSERT_NE(lattice, nullptr);
    auto expected = Interval::Degenerate(field, field.felt(i == 0 ? 7 : 9));
    const auto &actual = lattice->getValue().getScalarValue().getInterval();
    EXPECT_TRUE(checkCond(expected, actual, expected == actual));
    auto readers =
        analysis->getReadResults().find(SourceRef(fn.getArgument(i), {SourceRefIndex(member)}));
    ASSERT_NE(readers, analysis->getReadResults().end());
    EXPECT_TRUE(llvm::any_of(readers->second, [&](const IntervalAnalysisLattice *reader) {
      return reader == lattice;
    }));
  }
}

} // namespace
