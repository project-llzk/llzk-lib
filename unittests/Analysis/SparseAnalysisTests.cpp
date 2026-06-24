//===-- SparseAnalysisTests.cpp - Tests for sparse analysis -----*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/SparseAnalysis.h"
#include "llzk/Dialect/Function/IR/Ops.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Casting.h>

#include <cstddef>
#include <gtest/gtest.h>
#include <optional>

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kConsumerName("test.consumer");
constexpr llvm::StringLiteral kDeadSinkName("test.dead_sink");
constexpr llvm::StringLiteral kDelayedProducerName("test.delayed_producer");
constexpr llvm::StringLiteral kFunctionCallName("function.call");
constexpr llvm::StringLiteral kLoadName("test.load");
constexpr llvm::StringLiteral kLiveSinkName("test.live_sink");
constexpr llvm::StringLiteral kRegionSinkName("test.region_sink");
constexpr llvm::StringLiteral kSinkName("test.sink");
constexpr llvm::StringLiteral kSourceName("test.source");
constexpr llvm::StringLiteral kStoreName("test.store");
constexpr unsigned kInitialLatticeValue = 0;
constexpr unsigned kEntryLatticeValue = 1;
constexpr unsigned kDelayedProducerReadyValue = 42;
constexpr unsigned kSourceValue = 13;

class TestSparseLattice : public llzk::dataflow::AbstractSparseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSparseLattice)

  using AbstractSparseLattice::AbstractSparseLattice;

  unsigned getValue() const { return value; }

  ChangeResult setValue(unsigned next) {
    if (value == next) {
      return ChangeResult::NoChange;
    }
    value = next;
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override { os << value; }

private:
  unsigned value = kInitialLatticeValue;
};

/// A tiny state used to force the delayed producer lattice to change only after
/// the zero-result sink has already been visited once.
class TriggerState : public AnalysisState {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TriggerState)

  using AnalysisState::AnalysisState;

  bool isReady() const { return ready; }

  ChangeResult setReady() {
    if (ready) {
      return ChangeResult::NoChange;
    }
    ready = true;
    return ChangeResult::Change;
  }

  void print(raw_ostream &os) const override { os << (ready ? "ready" : "not ready"); }

private:
  bool ready = false;
};

class ZeroResultRevisitAnalysis
    : public llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice> {
  using Base = llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice>;

public:
  using Base::Base;

  LogicalResult initialize(Operation *top) override {
    topOp = top;
    return Base::initialize(top);
  }

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const TestSparseLattice *> operands,
      ArrayRef<TestSparseLattice *> results
  ) override {
    StringRef opName = op->getName().getStringRef();

    if (opName == kDelayedProducerName) {
      // Make the producer depend on the trigger. The first producer visit
      // leaves the result lattice at its optimistic value. After the sink sets
      // the trigger, the producer is revisited and changes its result lattice.
      TriggerState *trigger = getTriggerState();
      addDependency(trigger, getProgramPointAfter(op));
      if (!results.empty() && trigger->isReady()) {
        propagateIfChanged(results.front(), results.front()->setValue(kDelayedProducerReadyValue));
      }
      return success();
    }

    if (opName == kSinkName) {
      ++sinkVisitCount;
      if (operands.empty()) {
        sawSinkWithoutOperand = true;
        return success();
      }
      sinkOperandValues.push_back(operands.front()->getValue());

      // Trigger a later producer result update. The adapter's use-def
      // subscription should then revisit this zero-result sink.
      TriggerState *trigger = getTriggerState();
      propagateIfChanged(trigger, trigger->setReady());
      return success();
    }

    return success();
  }

  llvm::ArrayRef<unsigned> getSinkOperandValues() const { return sinkOperandValues; }
  unsigned getSinkVisitCount() const { return sinkVisitCount; }
  bool visitedSinkWithoutOperand() const { return sawSinkWithoutOperand; }

protected:
  void setToEntryState(TestSparseLattice *lattice) override {
    (void)lattice->setValue(kEntryLatticeValue);
  }

private:
  TriggerState *getTriggerState() {
    return getOrCreate<TriggerState>(getProgramPointBefore(topOp));
  }

  Operation *topOp = nullptr;
  unsigned sinkVisitCount = 0;
  bool sawSinkWithoutOperand = false;
  llvm::SmallVector<unsigned, 4> sinkOperandValues;
};

class ZeroResultProgramOrderAnalysis
    : public llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice> {
  using Base = llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice>;

public:
  using Base::Base;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const TestSparseLattice *> operands,
      ArrayRef<TestSparseLattice *> results
  ) override {
    StringRef opName = op->getName().getStringRef();

    if (opName == kSourceName) {
      if (!results.empty()) {
        propagateIfChanged(results.front(), results.front()->setValue(kSourceValue));
      }
      return success();
    }

    if (opName == kStoreName) {
      if (operands.empty()) {
        sawStoreWithoutOperand = true;
        return success();
      }
      storedValue = operands.front()->getValue();
      return success();
    }

    if (opName == kLoadName) {
      if (!results.empty() && storedValue.has_value()) {
        propagateIfChanged(results.front(), results.front()->setValue(*storedValue));
      }
      return success();
    }

    if (opName == kConsumerName) {
      if (operands.empty()) {
        sawConsumerWithoutOperand = true;
        return success();
      }
      consumerOperandValues.push_back(operands.front()->getValue());
      return success();
    }

    return success();
  }

  llvm::ArrayRef<unsigned> getConsumerOperandValues() const { return consumerOperandValues; }
  bool visitedConsumerWithoutOperand() const { return sawConsumerWithoutOperand; }
  bool visitedStoreWithoutOperand() const { return sawStoreWithoutOperand; }

protected:
  void setToEntryState(TestSparseLattice *lattice) override {
    (void)lattice->setValue(kEntryLatticeValue);
  }

private:
  std::optional<unsigned> storedValue;
  bool sawConsumerWithoutOperand = false;
  bool sawStoreWithoutOperand = false;
  llvm::SmallVector<unsigned, 4> consumerOperandValues;
};

class RegionBranchSkipAnalysis
    : public llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice> {
  using Base = llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice>;

public:
  using Base::Base;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const TestSparseLattice *> /*operands*/,
      ArrayRef<TestSparseLattice *> /*results*/
  ) override {
    if (llvm::isa<mlir::scf::IfOp>(op)) {
      sawIfTransfer = true;
    }
    if (op->getName().getStringRef() == kRegionSinkName) {
      sawRegionSink = true;
    }
    return success();
  }

  bool sawIfOperationTransfer() const { return sawIfTransfer; }
  bool sawRegionSinkTransfer() const { return sawRegionSink; }

protected:
  void setToEntryState(TestSparseLattice *lattice) override {
    (void)lattice->setValue(kEntryLatticeValue);
  }

private:
  bool sawIfTransfer = false;
  bool sawRegionSink = false;
};

class ZeroResultLivenessAnalysis
    : public llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice> {
  using Base = llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice>;

public:
  using Base::Base;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const TestSparseLattice *> /*operands*/,
      ArrayRef<TestSparseLattice *> /*results*/
  ) override {
    StringRef opName = op->getName().getStringRef();
    if (opName == kDeadSinkName) {
      ++deadSinkTransfers;
    } else if (opName == kLiveSinkName) {
      ++liveSinkTransfers;
    }
    return success();
  }

  unsigned getDeadSinkTransferCount() const { return deadSinkTransfers; }
  unsigned getLiveSinkTransferCount() const { return liveSinkTransfers; }

protected:
  void setToEntryState(TestSparseLattice *lattice) override {
    (void)lattice->setValue(kEntryLatticeValue);
  }

private:
  unsigned deadSinkTransfers = 0;
  unsigned liveSinkTransfers = 0;
};

class ZeroResultCallHandlingAnalysis
    : public llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice> {
  using Base = llzk::dataflow::SparseForwardDataFlowAnalysis<TestSparseLattice>;

public:
  using Base::Base;

  LogicalResult visitOperation(
      Operation *op, ArrayRef<const TestSparseLattice *> /*operands*/,
      ArrayRef<TestSparseLattice *> /*results*/
  ) override {
    if (isa<llzk::function::CallOp>(op)) {
      ++typedCallTransfers;
    }
    return success();
  }

  void visitExternalCall(
      CallOpInterface call, ArrayRef<const TestSparseLattice *> argumentLattices,
      ArrayRef<TestSparseLattice *> resultLattices
  ) override {
    if (call->getName().getStringRef() != kFunctionCallName) {
      return;
    }
    ++externalCallTransfers;
    externalArgumentCounts.push_back(argumentLattices.size());
    externalResultCounts.push_back(resultLattices.size());
  }

  unsigned getTypedCallTransferCount() const { return typedCallTransfers; }
  unsigned getExternalCallTransferCount() const { return externalCallTransfers; }
  llvm::ArrayRef<std::size_t> getExternalArgumentCounts() const { return externalArgumentCounts; }
  llvm::ArrayRef<std::size_t> getExternalResultCounts() const { return externalResultCounts; }

protected:
  void setToEntryState(TestSparseLattice *lattice) override {
    (void)lattice->setValue(kEntryLatticeValue);
  }

private:
  unsigned typedCallTransfers = 0;
  unsigned externalCallTransfers = 0;
  llvm::SmallVector<std::size_t, 4> externalArgumentCounts;
  llvm::SmallVector<std::size_t, 4> externalResultCounts;
};

class SparseAnalysisTests : public LLZKTest {};

Operation *findSingleZeroResultFunctionCall(ModuleOp module) {
  Operation *callOp = nullptr;
  WalkResult walkResult = module.walk([&](Operation *op) {
    if (!isa<llzk::function::CallOp>(op) || op->getNumResults() != 0) {
      return WalkResult::advance();
    }
    if (callOp != nullptr) {
      return WalkResult::interrupt();
    }
    callOp = op;
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? nullptr : callOp;
}

TEST_F(SparseAnalysisTests, RevisitsLiveZeroResultOpWhenOperandLatticeChanges) {
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module {
  %0 = "test.delayed_producer"() : () -> i32
  "test.sink"(%0) : (i32) -> ()
}
)mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  DataFlowSolver solver;
  llzk::dataflow::loadRequiredAnalyses(solver);
  auto *analysis = solver.load<ZeroResultRevisitAnalysis>();

  ASSERT_TRUE(succeeded(solver.initializeAndRun(module->getOperation())));

  llvm::ArrayRef<unsigned> values = analysis->getSinkOperandValues();
  EXPECT_FALSE(analysis->visitedSinkWithoutOperand());
  ASSERT_GE(analysis->getSinkVisitCount(), 2U);
  ASSERT_GE(values.size(), 2U);
  EXPECT_EQ(values.front(), kInitialLatticeValue);
  EXPECT_EQ(values.back(), kDelayedProducerReadyValue);
}

TEST_F(SparseAnalysisTests, InitializesZeroResultOpsInProgramOrder) {
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module {
  %0 = "test.source"() : () -> i32
  "test.store"(%0) : (i32) -> ()
  %1 = "test.load"() : () -> i32
  "test.consumer"(%1) : (i32) -> ()
}
)mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  DataFlowSolver solver;
  llzk::dataflow::loadRequiredAnalyses(solver);
  auto *analysis = solver.load<ZeroResultProgramOrderAnalysis>();

  ASSERT_TRUE(succeeded(solver.initializeAndRun(module->getOperation())));

  llvm::ArrayRef<unsigned> values = analysis->getConsumerOperandValues();
  EXPECT_FALSE(analysis->visitedStoreWithoutOperand());
  EXPECT_FALSE(analysis->visitedConsumerWithoutOperand());
  ASSERT_FALSE(values.empty());
  EXPECT_EQ(values.back(), kSourceValue);
}

TEST_F(SparseAnalysisTests, DoesNotCallTypedTransferForZeroResultRegionBranchOps) {
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module {
  %true = arith.constant true
  scf.if %true {
    "test.region_sink"() : () -> ()
  }
}
)mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  DataFlowSolver solver;
  llzk::dataflow::loadRequiredAnalyses(solver);
  auto *analysis = solver.load<RegionBranchSkipAnalysis>();

  ASSERT_TRUE(succeeded(solver.initializeAndRun(module->getOperation())));

  EXPECT_FALSE(analysis->sawIfOperationTransfer());
  EXPECT_TRUE(analysis->sawRegionSinkTransfer());
}

TEST_F(SparseAnalysisTests, DoesNotVisitZeroResultOpsInDeadBlocks) {
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module {
  %false = arith.constant false
  scf.if %false {
    "test.dead_sink"() : () -> ()
  }
  "test.live_sink"() : () -> ()
}
)mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  DataFlowSolver solver;
  llzk::dataflow::loadRequiredAnalyses(solver);
  auto *analysis = solver.load<ZeroResultLivenessAnalysis>();

  ASSERT_TRUE(succeeded(solver.initializeAndRun(module->getOperation())));

  EXPECT_EQ(analysis->getDeadSinkTransferCount(), 0U);
  EXPECT_GE(analysis->getLiveSinkTransferCount(), 1U);
}

TEST_F(SparseAnalysisTests, RoutesZeroResultExternalCallsThroughExternalCallHook) {
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module attributes {llzk.lang} {
  function.def @callee(%arg0: !felt.type) {
    function.return
  }
  function.def @caller(%arg0: !felt.type) {
    function.call @callee(%arg0) : (!felt.type) -> ()
    function.return
  }
}
)mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  DataFlowConfig config;
  config.setInterprocedural(false);
  DataFlowSolver solver(config);
  llzk::dataflow::loadRequiredAnalyses(solver);
  auto *analysis = solver.load<ZeroResultCallHandlingAnalysis>();

  ASSERT_TRUE(succeeded(solver.initializeAndRun(module->getOperation())));

  EXPECT_EQ(analysis->getTypedCallTransferCount(), 0U);
  EXPECT_GE(analysis->getExternalCallTransferCount(), 1U);

  bool sawExpectedCall = false;
  auto argCounts = analysis->getExternalArgumentCounts();
  auto resultCounts = analysis->getExternalResultCounts();
  ASSERT_EQ(argCounts.size(), resultCounts.size());
  for (auto [argCount, resultCount] : llvm::zip(argCounts, resultCounts)) {
    sawExpectedCall |= argCount == 1 && resultCount == 0;
  }
  EXPECT_TRUE(sawExpectedCall);
}

TEST_F(SparseAnalysisTests, SkipsTypedTransferForInterproceduralZeroResultCalls) {
  ctx.allowUnregisteredDialects();

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(
      R"mlir(
module attributes {llzk.lang} {
  function.def @callee(%arg0: !felt.type) {
    function.return
  }
  function.def @caller(%arg0: !felt.type) {
    function.call @callee(%arg0) : (!felt.type) -> ()
    function.return
  }
}
)mlir",
      &ctx
  );
  ASSERT_TRUE(module);

  Operation *callOp = findSingleZeroResultFunctionCall(*module);
  ASSERT_NE(callOp, nullptr);

  DataFlowSolver solver;
  llzk::dataflow::loadRequiredAnalyses(solver);
  auto *analysis = solver.load<ZeroResultCallHandlingAnalysis>();

  ASSERT_TRUE(succeeded(solver.initializeAndRun(module->getOperation())));

  EXPECT_EQ(analysis->getTypedCallTransferCount(), 0U);
  EXPECT_EQ(analysis->getExternalCallTransferCount(), 0U);

  auto *afterCall = solver.getProgramPointAfter(callOp);
  const auto *predecessors = solver.lookupState<mlir::dataflow::PredecessorState>(afterCall);
  ASSERT_NE(predecessors, nullptr);
}

} // namespace
