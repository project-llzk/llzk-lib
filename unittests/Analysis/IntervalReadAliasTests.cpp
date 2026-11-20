//===-- IntervalReadAliasTests.cpp - Nested read alias regressions -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Dialect/Bool/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;

namespace {

/// Check interval restrictions and SMT equalities for reads along structured control flow.
class IntervalReadAliasTests : public LLZKTest {
protected:
  /// Instantiate a nested read scenario and query its asserted comparison constraints.
  void
  checkAliases(StringRef body, bool memberReads, bool sameIndex, bool constants, bool shouldAlias) {
    SCOPED_TRACE(memberReads ? "component members" : "array elements");
    SCOPED_TRACE(sameIndex ? "same index" : "independent indices");
    SCOPED_TRACE(constants ? "conflicting constants" : "inequality");
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
  function.def @opaque(%array: !array.type<2 x !felt.type>, %cells: !array.type<2 x !struct.type<@Cell>>, %i: index, %input: !felt.type, %replacement: !struct.type<@Cell>) {
    array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type
    array.write %cells[%i] = %replacement : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    function.return
  }
  function.def @test(%array: !array.type<2 x !felt.type>, %cells: !array.type<2 x !struct.type<@Cell>>, %i: index, %j: index, %condition: i1, %input: !felt.type, %replacement: !struct.type<@Cell>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    %seven = felt.const 7
    %nine = felt.const 9
    // LEFT_READ
    // LEFT_CONSTRAINT
    // BODY
    function.return
  }
}
)mlir";
    auto replace = [&](StringRef marker, StringRef replacement) {
      for (size_t pos; (pos = source.find(marker)) != std::string::npos;) {
        source.replace(pos, marker.size(), replacement.data(), replacement.size());
      }
    };
    replace("// BODY", body);
    replace(
        "// LEFT_READ", memberReads ? R"mlir(
    %leftCell = array.read %cells[%i] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %left = struct.readm %leftCell[@value] : !struct.type<@Cell>, !felt.type
)mlir"
                                    : R"mlir(
    %left = array.read %array[%i] : !array.type<2 x !felt.type>, !felt.type
)mlir"
    );
    replace(
        "// READ", memberReads ? R"mlir(
    %rightCell = array.read %cells[RIGHT_INDEX] : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
    %right = struct.readm %rightCell[@value] : !struct.type<@Cell>, !felt.type
    // RIGHT_CONSTRAINT
)mlir"
                               : R"mlir(
    %right = array.read %array[RIGHT_INDEX] : !array.type<2 x !felt.type>, !felt.type
    // RIGHT_CONSTRAINT
)mlir"
    );
    replace("RIGHT_INDEX", sameIndex ? "%i" : "%j");
    replace(
        "// LEFT_CONSTRAINT", constants ? R"mlir(
    %leftCheck = bool.cmp eq(%left, %seven)
    bool.assert %leftCheck
)mlir"
                                        : ""
    );
    replace(
        "// RIGHT_CONSTRAINT", constants ? R"mlir(
    %rightCheck = bool.cmp eq(%right, %nine)
    bool.assert %rightCheck
)mlir"
                                         : R"mlir(
    %different = bool.cmp ne(%left, %right)
    bool.assert %different
)mlir"
    );
    replace(
        "// WRITE", memberReads ? R"mlir(
    array.write %cells[%i] = %replacement : !array.type<2 x !struct.type<@Cell>>, !struct.type<@Cell>
)mlir"
                                : R"mlir(
    array.write %array[%i] = %input : !array.type<2 x !felt.type>, !felt.type
)mlir"
    );
    replace("// CALL", R"mlir(
    function.call @opaque(%array, %cells, %i, %input, %replacement) : (!array.type<2 x !felt.type>, !array.type<2 x !struct.type<@Cell>>, index, !felt.type, !struct.type<@Cell>) -> ()
)mlir");
    auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
    ASSERT_TRUE(mod);
    const Field &field = Field::getField("babybear");
    auto smt = llvm::CreateZ3Solver();
    DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
    llzk::dataflow::loadRequiredAnalyses(solver);
    solver.load<SourceRefAnalysis>();
    solver.load<IntervalDataFlowAnalysis>(smt, field, false, false);
    ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));
    auto fn = mod->lookupSymbol<function::FuncDefOp>("test");
    SmallVector<boolean::CmpOp> comparisons;
    fn.walk([&](boolean::CmpOp comparison) { comparisons.push_back(comparison); });
    ASSERT_EQ(comparisons.size(), constants ? 2U : 1U);
    for (auto comparison : comparisons) {
      const auto *lattice = solver.lookupState<IntervalAnalysisLattice>(comparison.getResult());
      ASSERT_NE(lattice, nullptr);
      ASSERT_FALSE(lattice->getConstraints().empty());
      for (const ExpressionValue &constraint : lattice->getConstraints()) {
        smt->addConstraint(constraint.getExpr());
      }
      if (constants && shouldAlias) {
        const auto *read = solver.lookupState<IntervalAnalysisLattice>(comparison.getLhs());
        ASSERT_NE(read, nullptr);
        EXPECT_EQ(read->getValue().getScalarValue().getInterval(), Interval::Empty(field));
      }
    }
    EXPECT_EQ(smt->check(), std::optional<bool>(!shouldAlias));
  }
};

// An outer read remains equal along a write-free path, including every loop iteration.
TEST_F(IntervalReadAliasTests, NestedReadAliasesShareConstraints) {
  const std::pair<StringRef, StringRef> cases[] = {
      {"if then", "scf.if %condition { // READ\n }"},
      {"if else", "scf.if %condition { } else { // READ\n }"},
      {"nested if", "scf.if %condition { scf.if %condition { // READ\n } }"},
      {"write in other branch", "scf.if %condition { // WRITE\n } else { // READ\n }"},
      {"write after branch read", "scf.if %condition { // READ\n // WRITE\n }"},
      {"for", "scf.for %iv = %zero to %two step %one { // READ\n }"},
      {"if in for", "scf.for %iv = %zero to %two step %one { scf.if %condition { // READ\n } }"},
      {"for in if", "scf.if %condition { scf.for %iv = %zero to %two step %one { // READ\n } }"},
      {"while before", R"mlir(
        scf.while : () -> () {
          // READ
          scf.condition(%condition)
        } do {
          scf.yield
        }
      )mlir"},
      {"while after", R"mlir(
        scf.while : () -> () {
          scf.condition(%condition)
        } do {
          // READ
          scf.yield
        }
      )mlir"},
      {"parallel", "scf.parallel (%iv) = (%zero) to (%two) step (%one) { // READ\n scf.reduce }"},
      {"forall", "scf.forall (%iv) in (%two) { // READ\n scf.forall.in_parallel { } }"},
      {"execute region", "scf.execute_region { // READ\n scf.yield }"},
      {"index switch", R"mlir(
        scf.index_switch %i
        case 0 {
          // READ
          scf.yield
        }
        default { scf.yield }
      )mlir"},
      {"intervening read-only region", R"mlir(
        scf.if %condition {
          %unused = array.read %array[%j] : !array.type<2 x !felt.type>, !felt.type
        }
        scf.if %condition { // READ
        }
      )mlir"}
  };
  for (auto [name, body] : cases) {
    SCOPED_TRACE(name.str());
    for (bool memberReads : {false, true}) {
      for (bool constants : {false, true}) {
        checkAliases(body, memberReads, true, constants, true);
      }
    }
  }
}

// Possible writes anywhere on a crossed loop's backedge invalidate the equality proof.
TEST_F(IntervalReadAliasTests, NestedReadAliasesRespectWriteBarriers) {
  const std::pair<StringRef, StringRef> cases[] = {
      {"outer block write", "// WRITE\n scf.if %condition { // READ\n }"},
      {"branch write", "scf.if %condition { // WRITE\n // READ\n }"},
      {"nested branch write", "scf.if %condition { // WRITE\n scf.if %condition { // READ\n } }"},
      {"intervening conditional write",
       "scf.if %condition { // WRITE\n } scf.if %condition { // READ\n }"},
      {"outer call", "// CALL\n scf.if %condition { // READ\n }"},
      {"nested call", "scf.if %condition { // CALL\n // READ\n }"},
      {"loop write before read", "scf.for %iv = %zero to %two step %one { // WRITE\n // READ\n }"},
      {"loop write after read", "scf.for %iv = %zero to %two step %one { // READ\n // WRITE\n }"},
      {"loop call after read", "scf.for %iv = %zero to %two step %one { // READ\n // CALL\n }"},
      {"nested loop backedge write", R"mlir(
        scf.for %iv = %zero to %two step %one {
          scf.if %condition { // READ
          }
          // WRITE
        }
      )mlir"},
      {"while other region write", R"mlir(
        scf.while : () -> () {
          // READ
          scf.condition(%condition)
        } do {
          // WRITE
          scf.yield
        }
      )mlir"},
      {"while before region write", R"mlir(
        scf.while : () -> () {
          // WRITE
          scf.condition(%condition)
        } do {
          // READ
          scf.yield
        }
      )mlir"}
  };
  for (auto [name, body] : cases) {
    SCOPED_TRACE(name.str());
    for (bool memberReads : {false, true}) {
      for (bool constants : {false, true}) {
        checkAliases(body, memberReads, true, constants, false);
      }
    }
  }
  for (bool memberReads : {false, true}) {
    for (bool constants : {false, true}) {
      checkAliases("scf.if %condition { // READ\n }", memberReads, false, constants, false);
    }
  }
}

} // namespace
