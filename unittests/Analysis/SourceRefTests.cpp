//===-- SourceRefTests.cpp - Unit tests for SourceRef analysis -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"
#include "../LLZKTestUtils.h"

#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Analysis/SourceRef.h"
#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/StreamHelper.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Parser/Parser.h>

#include <gtest/gtest.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

class SourceRefTests : public LLZKTest {
protected:
  static constexpr auto kModule = R"mlir(
module attributes {llzk.lang} {
  struct.def @SourceRefs {
    struct.member @storage : !pod.type<[@value: !felt.type]>
    struct.member @other : !felt.type

    function.def @compute() -> !struct.type<@SourceRefs> {
      %self = struct.new : !struct.type<@SourceRefs>
      %temporary = struct.new : !struct.type<@SourceRefs>
      %pod = pod.new : !pod.type<[@storage: !felt.type]>
      function.return %self : !struct.type<@SourceRefs>
    }

    function.def @constrain(%self: !struct.type<@SourceRefs>) {
      function.return
    }
  }
}
)mlir";
};

TEST_F(SourceRefTests, IndexHalfOpenOverlap) {
  SourceRefIndex range(APInt(64, 2), APInt(64, 5));
  SourceRefIndex overlappingRange(APInt(64, 4), APInt(64, 7));
  SourceRefIndex adjacentRange(APInt(64, 5), APInt(64, 8));

  EXPECT_FALSE(range.overlaps(SourceRefIndex(1)));
  EXPECT_TRUE(range.overlaps(SourceRefIndex(2)));
  EXPECT_TRUE(range.overlaps(SourceRefIndex(4)));
  EXPECT_FALSE(range.overlaps(SourceRefIndex(5)));
  EXPECT_TRUE(range.overlaps(overlappingRange));
  EXPECT_FALSE(range.overlaps(adjacentRange));
}

TEST_F(SourceRefTests, MemberOrderingUsesNamesToBreakEqualLocations) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto members = llvm::to_vector(structDef.getOps<MemberDefOp>());
  ASSERT_EQ(members.size(), 2);
  members[0]->setLoc(FileLineColLoc::get(&ctx, "same.llzk", 1, 1));
  members[1]->setLoc(FileLineColLoc::get(&ctx, "same.llzk", 1, 1));

  auto forwardLocation = isLocationLess(members[0], members[1]);
  auto reverseLocation = isLocationLess(members[1], members[0]);
  ASSERT_TRUE(succeeded(forwardLocation));
  ASSERT_TRUE(succeeded(reverseLocation));
  EXPECT_FALSE(*forwardLocation);
  EXPECT_FALSE(*reverseLocation);

  EXPECT_TRUE(NamedOpLocationLess<MemberDefOp> {}(members[1], members[0]));
  EXPECT_FALSE(NamedOpLocationLess<MemberDefOp> {}(members[0], members[1]));
}

TEST_F(SourceRefTests, LatticePrefixReplacementPreservesUnmatchedRefs) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  pod::NewPodOp newPod;
  computeFn.walk([&](pod::NewPodOp op) { newPod = op; });
  ASSERT_TRUE(newPod);

  SourceRef computeRoot(mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  SourceRef constrainRoot(mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()));
  SourceRef computeMember(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storage)}
  );
  SourceRef expectedMember(
      mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()), {SourceRefIndex(storage)}
  );
  SourceRef unrelated(mlir::cast<OpResult>(newPod.getResult()));

  SourceRefLatticeValue value;
  EXPECT_EQ(value.insert(computeMember), ChangeResult::Change);
  EXPECT_EQ(value.insert(unrelated), ChangeResult::Change);
  TranslationMap replacements {{computeRoot, SourceRefLatticeValue(constrainRoot)}};
  auto [replaced, changed] = value.replacePrefixes(replacements);

  EXPECT_EQ(changed, ChangeResult::Change);
  EXPECT_TRUE(replaced.getScalarValue().contains(expectedMember));
  EXPECT_TRUE(replaced.getScalarValue().contains(unrelated));
  EXPECT_FALSE(replaced.getScalarValue().contains(computeMember));

  auto [translated, translatedChanged] = value.translate(replacements);
  EXPECT_EQ(translatedChanged, ChangeResult::Change);
  EXPECT_TRUE(translated.getScalarValue().contains(expectedMember));
  EXPECT_FALSE(translated.getScalarValue().contains(unrelated));
}

TEST_F(SourceRefTests, LatticeWritesPointsSubarraysAndRanges) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  SourceRef computeRoot(mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  SourceRef constrainRoot(mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()));

  SourceRefLatticeValue matrix(llvm::ArrayRef<int64_t>({2, 2}));
  EXPECT_EQ(
      matrix.write(
          {SourceRefIndex(APInt(64, 0)), SourceRefIndex(APInt(64, 1))},
          SourceRefLatticeValue(computeRoot)
      ),
      ChangeResult::Change
  );
  auto point = matrix.extract({SourceRefIndex(APInt(64, 0)), SourceRefIndex(APInt(64, 1))});
  ASSERT_TRUE(succeeded(point));
  EXPECT_EQ(point->first.getSingleValue(), computeRoot);

  SourceRefLatticeValue row(llvm::ArrayRef<int64_t>({2}));
  EXPECT_EQ(
      row.getElemFlatIdx(0).setValue(SourceRefLatticeValue(constrainRoot)), ChangeResult::Change
  );
  EXPECT_EQ(
      row.getElemFlatIdx(1).setValue(SourceRefLatticeValue(computeRoot)), ChangeResult::Change
  );
  EXPECT_EQ(matrix.write({SourceRefIndex(APInt(64, 1))}, row), ChangeResult::Change);
  auto writtenRow = matrix.extract({SourceRefIndex(APInt(64, 1))});
  ASSERT_TRUE(succeeded(writtenRow));
  ASSERT_TRUE(writtenRow->first.isArray());
  EXPECT_EQ(writtenRow->first.getElemFlatIdx(0).getSingleValue(), constrainRoot);
  EXPECT_EQ(writtenRow->first.getElemFlatIdx(1).getSingleValue(), computeRoot);

  SourceRefLatticeValue vector(llvm::ArrayRef<int64_t>({3}));
  EXPECT_EQ(
      vector.write(
          {SourceRefIndex(APInt(64, 1), APInt(64, 3))}, SourceRefLatticeValue(constrainRoot)
      ),
      ChangeResult::Change
  );
  for (uint64_t index = 1; index < 3; ++index) {
    auto ranged = vector.extract({SourceRefIndex(APInt(64, index))});
    ASSERT_TRUE(succeeded(ranged));
    EXPECT_TRUE(ranged->first.getScalarValue().contains(constrainRoot));
  }

  SourceRefLatticeValue tensor(llvm::ArrayRef<int64_t>({2, 2, 3}));
  SourceRefLatticeValue matrixSlice(llvm::ArrayRef<int64_t>({2, 3}));
  EXPECT_EQ(
      matrixSlice.getElemFlatIdx(0).setValue(SourceRefLatticeValue(computeRoot)),
      ChangeResult::Change
  );
  EXPECT_EQ(tensor.write({SourceRefIndex(APInt(64, 1))}, matrixSlice), ChangeResult::Change);

  SourceRefLatticeValue transposedSlice(llvm::ArrayRef<int64_t>({3, 2}));
  EXPECT_DEATH(
      (void)tensor.write({SourceRefIndex(APInt(64, 0))}, transposedSlice),
      "SourceRef array write value shape does not match selected storage"
  );
}

TEST_F(SourceRefTests, OnlyReturnedComputeStructOverlapsConstrainSelf) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  auto allocations = llvm::to_vector(computeFn.getOps<CreateStructOp>());
  ASSERT_EQ(allocations.size(), 2);

  Value returnedSelf = computeFn.getSelfValueFromCompute();
  CreateStructOp temporary =
      allocations[0].getResult() == returnedSelf ? allocations[1] : allocations[0];
  auto constrainSelf = mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  SourceRef computeMember(mlir::cast<OpResult>(returnedSelf), {SourceRefIndex(storage)});
  SourceRef temporaryMember(mlir::cast<OpResult>(temporary.getResult()), {SourceRefIndex(storage)});
  SourceRef constrainMember(constrainSelf, {SourceRefIndex(storage)});

  EXPECT_TRUE(computeMember.overlaps(constrainMember));
  EXPECT_TRUE(constrainMember.overlaps(computeMember));
  EXPECT_FALSE(temporaryMember.overlaps(computeMember));
  EXPECT_FALSE(computeMember.overlaps(temporaryMember));
  EXPECT_FALSE(temporaryMember.overlaps(constrainMember));
  EXPECT_FALSE(constrainMember.overlaps(temporaryMember));
}

TEST_F(SourceRefTests, OnlyConstrainEntryArgumentOverlapsComputeSelf) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  auto constrainSelf = mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  auto *successor = new Block();
  constrainFn.getBody().push_back(successor);
  auto successorArg = successor->addArgument(constrainSelf.getType(), loc);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(successor);
  builder.create<llzk::function::ReturnOp>(loc);

  SourceRef computeMember(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storage)}
  );
  SourceRef constrainMember(constrainSelf, {SourceRefIndex(storage)});
  SourceRef successorMember(successorArg, {SourceRefIndex(storage)});

  EXPECT_TRUE(computeMember.overlaps(constrainMember));
  EXPECT_TRUE(constrainMember.overlaps(computeMember));
  EXPECT_FALSE(successorMember.overlaps(computeMember));
  EXPECT_FALSE(computeMember.overlaps(successorMember));
  EXPECT_FALSE(successorMember.overlaps(constrainMember));
  EXPECT_FALSE(constrainMember.overlaps(successorMember));
}

TEST_F(SourceRefTests, PodRecordsAndMembersRemainDistinct) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  pod::NewPodOp newPod;
  computeFn.walk([&](pod::NewPodOp op) { newPod = op; });
  ASSERT_TRUE(newPod);
  SourceRef root(mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  SourceRef memberRef(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storage)}
  );
  SourceRef podRef(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()),
      {SourceRefIndex(StringAttr::get(&ctx, "storage"))}
  );

  EXPECT_FALSE(memberRef.isValidPrefix(podRef));
  EXPECT_FALSE(podRef.isValidPrefix(memberRef));
  EXPECT_FALSE(memberRef.overlaps(podRef));
  EXPECT_FALSE(podRef.overlaps(memberRef));
  EXPECT_TRUE(memberRef.isValidPrefix(root));

  SourceRef arbitraryPodRef(
      mlir::cast<OpResult>(newPod.getResult()), {SourceRefIndex(StringAttr::get(&ctx, "storage"))}
  );
  EXPECT_FALSE(memberRef.isValidPrefix(arbitraryPodRef));
  EXPECT_FALSE(memberRef.overlaps(arbitraryPodRef));
}

TEST_F(SourceRefTests, StorageDependenciesAreSeparateFromAddressIdentity) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @StorageDependencies {
    struct.member @storage : !pod.type<[@value: !felt.type, @missing: !felt.type]>

    function.def @compute(%initial: !felt.type, %replacement: !felt.type)
        -> !struct.type<@StorageDependencies> {
      %self = struct.new : !struct.type<@StorageDependencies>
      %storage = pod.new { @value = %initial }
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>
      pod.write %storage[@value] = %replacement
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      %written = pod.read %storage[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type

      %other = pod.new { @value = %written }
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>
      %transitive = pod.read %other[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      %unwritten = pod.read %storage[@missing]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type

      %conditional = pod.new { @value = %initial }
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>
      %true = arith.constant true
      scf.if %true {
        pod.write %conditional[@value] = %replacement
            : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      }
      %maybe = pod.read %conditional[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type

      %cycleA = pod.new : !pod.type<[@value: !felt.type, @missing: !felt.type]>
      %cycleB = pod.new : !pod.type<[@value: !felt.type, @missing: !felt.type]>
      %cycleBValue = pod.read %cycleB[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      pod.write %cycleA[@value] = %cycleBValue
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      %cycleAValue = pod.read %cycleA[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      pod.write %cycleB[@value] = %cycleAValue
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      %cycle = pod.read %cycleA[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type

      struct.writem %self[@storage] = %storage
          : !struct.type<@StorageDependencies>,
            !pod.type<[@value: !felt.type, @missing: !felt.type]>
      function.return %self : !struct.type<@StorageDependencies>
    }

    function.def @constrain(
        %self: !struct.type<@StorageDependencies>,
        %initial: !felt.type,
        %replacement: !felt.type
    ) {
      %storage = struct.readm %self[@storage]
          : !struct.type<@StorageDependencies>,
            !pod.type<[@value: !felt.type, @missing: !felt.type]>
      %value = pod.read %storage[@value]
          : !pod.type<[@value: !felt.type, @missing: !felt.type]>, !felt.type
      constrain.eq %value, %replacement : !felt.type
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storageMember = *structDef.getOps<MemberDefOp>().begin();

  llvm::SmallVector<pod::NewPodOp> pods;
  llvm::SmallVector<pod::ReadPodOp> reads;
  component::MemberWriteOp storageWrite;
  computeFn.walk([&](pod::NewPodOp op) { pods.push_back(op); });
  computeFn.walk([&](pod::ReadPodOp op) { reads.push_back(op); });
  computeFn.walk([&](component::MemberWriteOp op) { storageWrite = op; });
  ASSERT_EQ(pods.size(), 5U);
  ASSERT_EQ(reads.size(), 7U);
  ASSERT_TRUE(storageWrite);
  pod::ReadPodOp constrainRead;
  constrainFn.walk([&](pod::ReadPodOp op) { constrainRead = op; });
  ASSERT_TRUE(constrainRead);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  DataFlowSolver &solver = analysis.getSolver();

  auto writeTarget = SourceRefAnalysis::getWriteTargetState(solver, storageWrite);
  ASSERT_TRUE(succeeded(writeTarget));
  SourceRef expectedWriteTarget(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storageMember)}
  );
  EXPECT_EQ(writeTarget->foldToScalar(), SourceRefSet({expectedWriteTarget}));

  SourceRef constrainStorageValue(
      mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()),
      {SourceRefIndex(storageMember), SourceRefIndex(StringAttr::get(&ctx, "value"))}
  );
  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(solver, constrainRead.getResult()).foldToScalar(),
      SourceRefSet({constrainStorageValue})
  );
  const ConstraintDependencyGraph &graph = analysis.getResult(structDef);
  SourceRef constrainStorage(
      mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()),
      {SourceRefIndex(storageMember)}
  );
  EXPECT_TRUE(
      graph.getConstrainingValues(constrainStorage).contains(SourceRef(constrainFn.getArgument(2)))
  );

  auto valueName = StringAttr::get(&ctx, "value");
  auto missingName = StringAttr::get(&ctx, "missing");
  SourceRef storageValue(mlir::cast<OpResult>(pods[0].getResult()), {SourceRefIndex(valueName)});
  SourceRef storageMissing(
      mlir::cast<OpResult>(pods[0].getResult()), {SourceRefIndex(missingName)}
  );
  SourceRef rebasedMissing(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()),
      {SourceRefIndex(storageMember), SourceRefIndex(missingName)}
  );
  SourceRef otherValue(mlir::cast<OpResult>(pods[1].getResult()), {SourceRefIndex(valueName)});

  auto rawWritten = SourceRefAnalysis::getValueState(solver, reads[0].getResult());
  ASSERT_TRUE(rawWritten.isSingleValue());
  EXPECT_EQ(rawWritten.getSingleValue(), storageValue);
  auto writtenDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[0].getResult()).foldToScalar();
  EXPECT_EQ(writtenDependencies, SourceRefSet({SourceRef(computeFn.getArgument(1))}));

  auto rawTransitive = SourceRefAnalysis::getValueState(solver, reads[1].getResult());
  ASSERT_TRUE(rawTransitive.isSingleValue());
  EXPECT_EQ(rawTransitive.getSingleValue(), otherValue);
  auto transitiveDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[1].getResult()).foldToScalar();
  EXPECT_EQ(transitiveDependencies, SourceRefSet({SourceRef(computeFn.getArgument(1))}));

  auto rawUnwritten = SourceRefAnalysis::getValueState(solver, reads[2].getResult());
  ASSERT_TRUE(rawUnwritten.isSingleValue());
  EXPECT_EQ(rawUnwritten.getSingleValue(), storageMissing);
  auto unwrittenDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[2].getResult()).foldToScalar();
  EXPECT_EQ(unwrittenDependencies, SourceRefSet({storageMissing}));

  auto aggregateDependencies =
      SourceRefAnalysis::getDependencyState(solver, storageWrite.getVal()).foldToScalar();
  EXPECT_EQ(
      aggregateDependencies, SourceRefSet({SourceRef(computeFn.getArgument(1)), rebasedMissing})
  );

  auto maybeDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[3].getResult()).foldToScalar();
  EXPECT_TRUE(maybeDependencies.contains(SourceRef(computeFn.getArgument(0))));
  EXPECT_TRUE(maybeDependencies.contains(SourceRef(computeFn.getArgument(1))));

  auto cycleDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[6].getResult()).foldToScalar();
  EXPECT_FALSE(cycleDependencies.empty());
}

TEST_F(SourceRefTests, ConditionalStorageWritePreservesNondeterministicAlternative) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @NondeterministicStorage {
    function.def @compute(%replacement: !felt.type) -> !struct.type<@NondeterministicStorage> {
      %self = struct.new : !struct.type<@NondeterministicStorage>
      %unknown = llzk.nondet : !felt.type
      %storage = pod.new { @value = %unknown } : !pod.type<[@value: !felt.type]>
      %true = arith.constant true
      scf.if %true {
        pod.write %storage[@value] = %replacement
            : !pod.type<[@value: !felt.type]>, !felt.type
      }
      %read = pod.read %storage[@value]
          : !pod.type<[@value: !felt.type]>, !felt.type
      function.return %self : !struct.type<@NondeterministicStorage>
    }

    function.def @constrain(
        %self: !struct.type<@NondeterministicStorage>, %replacement: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto nondet = *computeFn.getOps<NonDetOp>().begin();
  auto read = *computeFn.getOps<pod::ReadPodOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  DataFlowSolver &solver = analysis.getSolver();

  SourceRefSet dependencies =
      SourceRefAnalysis::getDependencyState(solver, read.getResult()).foldToScalar();
  EXPECT_TRUE(dependencies.contains(SourceRef(mlir::cast<OpResult>(nondet.getResult()))));
  EXPECT_TRUE(dependencies.contains(SourceRef(computeFn.getArgument(0))));
}

TEST_F(SourceRefTests, DefiniteStorageWritesFollowProgramOrderAfterSolverRevisit) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @OrderedStorage {
    function.def @compute(
        %earlier: !felt.type, %alternative: !felt.type, %final: !felt.type
    ) -> !struct.type<@OrderedStorage> {
      %self = struct.new : !struct.type<@OrderedStorage>
      %storage = pod.new : !pod.type<[@value: !felt.type]>
      %true = arith.constant true
      %selected = scf.if %true -> (!felt.type) {
        scf.yield %earlier : !felt.type
      } else {
        scf.yield %alternative : !felt.type
      }
      pod.write %storage[@value] = %selected
          : !pod.type<[@value: !felt.type]>, !felt.type
      pod.write %storage[@value] = %final
          : !pod.type<[@value: !felt.type]>, !felt.type
      %read = pod.read %storage[@value]
          : !pod.type<[@value: !felt.type]>, !felt.type
      function.return %self : !struct.type<@OrderedStorage>
    }

    function.def @constrain(
        %self: !struct.type<@OrderedStorage>, %earlier: !felt.type,
        %alternative: !felt.type, %final: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto read = *computeFn.getOps<pod::ReadPodOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  DataFlowSolver &solver = analysis.getSolver();

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(solver, read.getResult()).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(2))})
  );
}

TEST_F(SourceRefTests, AggregatePodInitializerRebasesNestedRecordDependencies) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @NestedPodStorage {
    function.def @compute(%left: !felt.type, %right: !felt.type)
        -> !struct.type<@NestedPodStorage> {
      %self = struct.new : !struct.type<@NestedPodStorage>
      %inner = pod.new { @left = %left, @right = %right }
          : !pod.type<[@left: !felt.type, @right: !felt.type]>
      %outer = pod.new { @nested = %inner }
          : !pod.type<[@nested: !pod.type<[@left: !felt.type, @right: !felt.type]>]>
      %nested = pod.read %outer[@nested]
          : !pod.type<[@nested: !pod.type<[@left: !felt.type, @right: !felt.type]>]>,
            !pod.type<[@left: !felt.type, @right: !felt.type]>
      %read = pod.read %nested[@left]
          : !pod.type<[@left: !felt.type, @right: !felt.type]>, !felt.type
      function.return %self : !struct.type<@NestedPodStorage>
    }

    function.def @constrain(
        %self: !struct.type<@NestedPodStorage>, %left: !felt.type, %right: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto reads = llvm::to_vector(computeFn.getOps<pod::ReadPodOp>());
  ASSERT_EQ(reads.size(), 2U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  DataFlowSolver &solver = analysis.getSolver();

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(solver, reads[1].getResult()).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
}

TEST_F(SourceRefTests, StorageReadsResolveAtTheirProgramPoint) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ReadBeforeWrite {
    function.def @compute(%initial: !felt.type, %replacement: !felt.type)
        -> !struct.type<@ReadBeforeWrite> {
      %self = struct.new : !struct.type<@ReadBeforeWrite>
      %storage = pod.new { @value = %initial } : !pod.type<[@value: !felt.type]>
      %before = pod.read %storage[@value]
          : !pod.type<[@value: !felt.type]>, !felt.type
      pod.write %storage[@value] = %replacement
          : !pod.type<[@value: !felt.type]>, !felt.type
      function.return %self : !struct.type<@ReadBeforeWrite>
    }

    function.def @constrain(
        %self: !struct.type<@ReadBeforeWrite>, %initial: !felt.type,
        %replacement: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto read = *computeFn.getOps<pod::ReadPodOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), read.getResult()).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
}

TEST_F(SourceRefTests, StorageDependenciesTranslateAcrossFunctionReturns) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  function.def @load(%input: !felt.type) -> !felt.type {
    %storage = pod.new { @value = %input } : !pod.type<[@value: !felt.type]>
    %read = pod.read %storage[@value]
        : !pod.type<[@value: !felt.type]>, !felt.type
    function.return %read : !felt.type
  }

  struct.def @CallStorage {
    function.def @compute(%input: !felt.type) -> !struct.type<@CallStorage> {
      %self = struct.new : !struct.type<@CallStorage>
      %loaded = function.call @load(%input) : (!felt.type) -> !felt.type
      function.return %self : !struct.type<@CallStorage>
    }

    function.def @constrain(%self: !struct.type<@CallStorage>, %input: !felt.type) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto call = *computeFn.getOps<function::CallOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), call.getResult(0)).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
}

TEST_F(SourceRefTests, StorageDependenciesTranslateIntoNestedConstrainCalls) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @Child {
    function.def @compute(%input: !felt.type) -> !struct.type<@Child> {
      %self = struct.new : !struct.type<@Child>
      function.return %self : !struct.type<@Child>
    }

    function.def @constrain(%self: !struct.type<@Child>, %input: !felt.type) {
      %zero = felt.const 0
      constrain.eq %input, %zero : !felt.type, !felt.type
      function.return
    }
  }

  struct.def @Parent {
    struct.member @child : !struct.type<@Child>

    function.def @compute(%input: !felt.type) -> !struct.type<@Parent> {
      %self = struct.new : !struct.type<@Parent>
      %child = function.call @Child::@compute(%input)
          : (!felt.type) -> !struct.type<@Child>
      struct.writem %self[@child] = %child
          : !struct.type<@Parent>, !struct.type<@Child>
      function.return %self : !struct.type<@Parent>
    }

    function.def @constrain(%self: !struct.type<@Parent>, %input: !felt.type) {
      %child = struct.readm %self[@child]
          : !struct.type<@Parent>, !struct.type<@Child>
      %storage = pod.new { @value = %input } : !pod.type<[@value: !felt.type]>
      %read = pod.read %storage[@value]
          : !pod.type<[@value: !felt.type]>, !felt.type
      function.call @Child::@constrain(%child, %read)
          : (!struct.type<@Child>, !felt.type) -> ()
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structs = llvm::to_vector(mod->getOps<StructDefOp>());
  ASSERT_EQ(structs.size(), 2U);
  auto childConstrain = structs[0].getConstrainFuncOp();
  auto parentConstrain = structs[1].getConstrainFuncOp();
  auto zero = *childConstrain.getOps<felt::FeltConstantOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  SourceRef parentInput(parentConstrain.getArgument(1));
  SourceRef zeroRef(zero);
  EXPECT_TRUE(analysis.getResult(structs[1]).getConstrainingValues(parentInput).contains(zeroRef));
}

TEST_F(SourceRefTests, ArrayPodInitializerRebasesElementDependencies) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ArrayPodStorage {
    function.def @compute(%left: !felt.type, %right: !felt.type)
        -> !struct.type<@ArrayPodStorage> {
      %self = struct.new : !struct.type<@ArrayPodStorage>
      %values = array.new %left, %right : !array.type<2 x !felt.type>
      %storage = pod.new { @values = %values }
          : !pod.type<[@values: !array.type<2 x !felt.type>]>
      %stored = pod.read %storage[@values]
          : !pod.type<[@values: !array.type<2 x !felt.type>]>,
            !array.type<2 x !felt.type>
      %c0 = arith.constant 0 : index
      %read = array.read %stored[%c0] : !array.type<2 x !felt.type>, !felt.type
      function.return %self : !struct.type<@ArrayPodStorage>
    }

    function.def @constrain(
        %self: !struct.type<@ArrayPodStorage>, %left: !felt.type, %right: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto read = *computeFn.getOps<array::ReadArrayOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), read.getResult()).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
}

TEST_F(SourceRefTests, ArrayMutationsUpdateStorageDependencies) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ArrayMutations {
    function.def @compute(%uninitializedValue: !felt.type, %initial: !felt.type,
                          %replacement: !felt.type) -> !struct.type<@ArrayMutations> {
      %self = struct.new : !struct.type<@ArrayMutations>
      %c0 = arith.constant 0 : index
      %uninitialized = array.new : !array.type<1 x !felt.type>
      array.write %uninitialized[%c0] = %uninitializedValue
          : !array.type<1 x !felt.type>, !felt.type
      %readUninitialized = array.read %uninitialized[%c0]
          : !array.type<1 x !felt.type>, !felt.type
      %initialized = array.new %initial : !array.type<1 x !felt.type>
      array.write %initialized[%c0] = %replacement
          : !array.type<1 x !felt.type>, !felt.type
      %readOverwritten = array.read %initialized[%c0]
          : !array.type<1 x !felt.type>, !felt.type
      function.return %self : !struct.type<@ArrayMutations>
    }

    function.def @constrain(
        %self: !struct.type<@ArrayMutations>, %uninitializedValue: !felt.type,
        %initial: !felt.type, %replacement: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto reads = llvm::to_vector(computeFn.getOps<array::ReadArrayOp>());
  ASSERT_EQ(reads.size(), 2U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  DataFlowSolver &solver = analysis.getSolver();

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(solver, reads[0].getResult()).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(solver, reads[1].getResult()).foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(2))})
  );
}

TEST_F(SourceRefTests, ConditionalAggregateAliasPreservesBothAlternatives) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ConditionalAggregateAlias {
    function.def @compute(%initial: !felt.type, %replacement: !felt.type)
        -> !struct.type<@ConditionalAggregateAlias> {
      %self = struct.new : !struct.type<@ConditionalAggregateAlias>
      %first = pod.new { @value = %initial } : !pod.type<[@value: !felt.type]>
      %storage = pod.new { @nested = %first }
          : !pod.type<[@nested: !pod.type<[@value: !felt.type]>]>
      %condition = arith.constant true
      scf.if %condition {
        %second = pod.new { @value = %replacement } : !pod.type<[@value: !felt.type]>
        pod.write %storage[@nested] = %second
            : !pod.type<[@nested: !pod.type<[@value: !felt.type]>]>,
              !pod.type<[@value: !felt.type]>
      }
      %nested = pod.read %storage[@nested]
          : !pod.type<[@nested: !pod.type<[@value: !felt.type]>]>,
            !pod.type<[@value: !felt.type]>
      %read = pod.read %nested[@value]
          : !pod.type<[@value: !felt.type]>, !felt.type
      function.return %self : !struct.type<@ConditionalAggregateAlias>
    }

    function.def @constrain(
        %self: !struct.type<@ConditionalAggregateAlias>, %initial: !felt.type,
        %replacement: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto reads = llvm::to_vector(computeFn.getOps<pod::ReadPodOp>());
  ASSERT_EQ(reads.size(), 2U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), reads[1].getResult())
          .foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0)), SourceRef(computeFn.getArgument(1))})
  );
}

TEST_F(SourceRefTests, AggregateArgumentInitializerPreservesChildPath) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @AggregateArgumentInitializer {
    function.def @compute(%source: !pod.type<[@left: !felt.type, @right: !felt.type]>)
        -> !struct.type<@AggregateArgumentInitializer> {
      %self = struct.new : !struct.type<@AggregateArgumentInitializer>
      %storage = pod.new { @nested = %source }
          : !pod.type<[@nested: !pod.type<[@left: !felt.type, @right: !felt.type]>]>
      %nested = pod.read %storage[@nested]
          : !pod.type<[@nested: !pod.type<[@left: !felt.type, @right: !felt.type]>]>,
            !pod.type<[@left: !felt.type, @right: !felt.type]>
      %read = pod.read %nested[@left]
          : !pod.type<[@left: !felt.type, @right: !felt.type]>, !felt.type
      function.return %self : !struct.type<@AggregateArgumentInitializer>
    }

    function.def @constrain(
        %self: !struct.type<@AggregateArgumentInitializer>,
        %source: !pod.type<[@left: !felt.type, @right: !felt.type]>
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto reads = llvm::to_vector(computeFn.getOps<pod::ReadPodOp>());
  ASSERT_EQ(reads.size(), 2U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  SourceRef expected(
      mlir::cast<BlockArgument>(computeFn.getArgument(0)),
      {SourceRefIndex(StringAttr::get(&ctx, "left"))}
  );
  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), reads[1].getResult())
          .foldToScalar(),
      SourceRefSet({expected})
  );
}

TEST_F(SourceRefTests, AggregateStructMemberInitializerPreservesChildPath) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @AggregateStructMemberInitializer {
    struct.member @source : !pod.type<[@left: !felt.type, @right: !felt.type]>

    function.def @compute(%left: !felt.type, %right: !felt.type)
        -> !struct.type<@AggregateStructMemberInitializer> {
      %self = struct.new : !struct.type<@AggregateStructMemberInitializer>
      %source = pod.new { @left = %left, @right = %right }
          : !pod.type<[@left: !felt.type, @right: !felt.type]>
      struct.writem %self[@source] = %source
          : !struct.type<@AggregateStructMemberInitializer>,
            !pod.type<[@left: !felt.type, @right: !felt.type]>
      %member = struct.readm %self[@source]
          : !struct.type<@AggregateStructMemberInitializer>,
            !pod.type<[@left: !felt.type, @right: !felt.type]>
      %storage = pod.new { @nested = %member }
          : !pod.type<[@nested: !pod.type<[@left: !felt.type, @right: !felt.type]>]>
      %nested = pod.read %storage[@nested]
          : !pod.type<[@nested: !pod.type<[@left: !felt.type, @right: !felt.type]>]>,
            !pod.type<[@left: !felt.type, @right: !felt.type]>
      %read = pod.read %nested[@left]
          : !pod.type<[@left: !felt.type, @right: !felt.type]>, !felt.type
      function.return %self : !struct.type<@AggregateStructMemberInitializer>
    }

    function.def @constrain(
        %self: !struct.type<@AggregateStructMemberInitializer>, %left: !felt.type,
        %right: !felt.type
    ) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto reads = llvm::to_vector(computeFn.getOps<pod::ReadPodOp>());
  ASSERT_EQ(reads.size(), 2U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), reads[1].getResult())
          .foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
}

TEST_F(SourceRefTests, ComputeSelfRebasesToConstrainSelfWithoutChangingPath) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  auto valueName = StringAttr::get(&ctx, "value");

  SourceRef computeSelf(mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  auto constrainSelfArg = mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());
  SourceRef constrainSelf(constrainSelfArg);
  SourceRef computeValue(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()),
      {SourceRefIndex(storage), SourceRefIndex(valueName)}
  );
  SourceRef expectedConstrainValue(
      constrainSelfArg, {SourceRefIndex(storage), SourceRefIndex(valueName)}
  );

  auto translated = computeValue.translate(computeSelf, constrainSelf);
  ASSERT_TRUE(succeeded(translated));
  EXPECT_EQ(*translated, expectedConstrainValue);

  SourceRef mismatchedComputeValue(
      mlir::cast<OpResult>(computeFn.getSelfValueFromCompute()),
      {SourceRefIndex(StringAttr::get(&ctx, "storage")), SourceRefIndex(valueName)}
  );
  auto mismatchedTranslation = mismatchedComputeValue.translate(computeSelf, constrainSelf);
  ASSERT_TRUE(succeeded(mismatchedTranslation));
  EXPECT_NE(*mismatchedTranslation, expectedConstrainValue);
  EXPECT_TRUE(mismatchedTranslation->getPath().front().isPodRecord());
}

TEST_F(SourceRefTests, OnlyConstrainEntryArgumentPrintsAsSelf) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  auto constrainSelf = mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  auto *successor = new Block();
  constrainFn.getBody().push_back(successor);
  auto successorArg = successor->addArgument(constrainSelf.getType(), loc);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(successor);
  builder.create<llzk::function::ReturnOp>(loc);

  EXPECT_EQ(buildStringViaPrint(SourceRef(constrainSelf)), "%self");
  EXPECT_EQ(
      buildStringViaPrint(SourceRef(constrainSelf, {SourceRefIndex(storage)})), "%self.storage"
  );
  EXPECT_EQ(buildStringViaPrint(SourceRef(successorArg)), "%arg0");
  EXPECT_EQ(
      buildStringViaPrint(SourceRef(successorArg, {SourceRefIndex(storage)})), "%arg0.storage"
  );
}

TEST_F(SourceRefTests, ImmutableGlobalIsNotATemplateConstant) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  global.def const @N : index = 3

  function.def @read_global() -> index {
    %value = global.read @N : index
    function.return %value : index
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto func = *mod->getOps<function::FuncDefOp>().begin();
  auto read = *func.getOps<global::GlobalReadOp>().begin();
  auto ref = SourceRefLattice::getSourceRef(read.getResult());
  ASSERT_TRUE(succeeded(ref));
  EXPECT_TRUE(ref->isRooted());
  EXPECT_FALSE(ref->isTemplateConstant());
  EXPECT_TRUE(ref->isImmutableGlobal());
}

TEST_F(SourceRefTests, ScfBlockArgumentsUseUnnamedFallback) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ScfBlockArg {
    function.def @compute() -> !struct.type<@ScfBlockArg> {
      %self = struct.new : !struct.type<@ScfBlockArg>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %loop = scf.while (%i = %c0) : (index) -> index {
        %cond = arith.cmpi slt, %i, %c1 : index
        scf.condition(%cond) %i : index
      } do {
      ^bb0(%i: index):
        %next = arith.addi %i, %c1 : index
        scf.yield %next : index
      }
      function.return %self : !struct.type<@ScfBlockArg>
    }

    function.def @constrain(%self: !struct.type<@ScfBlockArg>) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  scf::WhileOp whileOp;
  mod->walk([&](scf::WhileOp op) { whileOp = op; });
  ASSERT_TRUE(whileOp);

  SourceRef afterArg(whileOp.getAfter().front().getArgument(0));
  EXPECT_EQ(buildStringViaPrint(afterArg), "%arg0");
}

TEST_F(SourceRefTests, ConstraintQueriesNarrowStaticLoopRangesPerElement) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @LoopRanges {
    struct.member @out : !array.type<3 x !felt.type> {llzk.pub, signal}

    function.def @compute(%in: !array.type<3 x !felt.type>)
        -> !struct.type<@LoopRanges> {
      %self = struct.new : !struct.type<@LoopRanges>
      %storage = array.new : !array.type<3 x !felt.type>
      struct.writem %self[@out] = %storage
          : !struct.type<@LoopRanges>, !array.type<3 x !felt.type>
      function.return %self : !struct.type<@LoopRanges>
    }

    function.def @constrain(
        %self: !struct.type<@LoopRanges>,
        %in: !array.type<3 x !felt.type>) {
      %out = struct.readm %self[@out]
          : !struct.type<@LoopRanges>, !array.type<3 x !felt.type>
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %lhs = array.read %out[%i] : !array.type<3 x !felt.type>, !felt.type
        %rhs = array.read %in[%i] : !array.type<3 x !felt.type>, !felt.type
        constrain.eq %lhs, %rhs : !felt.type
      }
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto outMember = *structDef.getOps<MemberDefOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  const ConstraintDependencyGraph &graph = analysis.getResult(structDef);

  for (uint64_t index = 0; index < 2; ++index) {
    SourceRef output(
        constrainFn.getArgument(0), {SourceRefIndex(outMember), SourceRefIndex(APInt(64, index))}
    );
    SourceRef input(constrainFn.getArgument(1), {SourceRefIndex(APInt(64, index))});
    EXPECT_TRUE(graph.getConstrainingValues(output).contains(input));
  }

  SourceRef omitted(
      constrainFn.getArgument(0), {SourceRefIndex(outMember), SourceRefIndex(APInt(64, 2))}
  );
  EXPECT_TRUE(graph.getConstrainingValues(omitted).empty());
}

TEST_F(SourceRefTests, ConstraintQueriesDropOnlyNeutralLoopZeroInitializers) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @LoopNormalization {
    struct.member @sum : !felt.type {llzk.pub, signal}
    struct.member @zero : !felt.type {llzk.pub, signal}

    function.def @compute(%in: !felt.type) -> !struct.type<@LoopNormalization> {
      %self = struct.new : !struct.type<@LoopNormalization>
      struct.writem %self[@sum] = %in : !struct.type<@LoopNormalization>, !felt.type
      struct.writem %self[@zero] = %in : !struct.type<@LoopNormalization>, !felt.type
      function.return %self : !struct.type<@LoopNormalization>
    }

    function.def @constrain(%self: !struct.type<@LoopNormalization>, %in: !felt.type)
        attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      %sum = struct.readm %self[@sum] : !struct.type<@LoopNormalization>, !felt.type
      %zeroOut = struct.readm %self[@zero] : !struct.type<@LoopNormalization>, !felt.type
      %zero = felt.const 0
      %true = arith.constant true
      %loop = scf.while (%acc = %zero) : (!felt.type) -> !felt.type {
        scf.condition(%true) %acc : !felt.type
      } do {
      ^bb0(%acc: !felt.type):
        %next = felt.add %acc, %in : !felt.type, !felt.type
        scf.yield %next : !felt.type
      }
      constrain.eq %sum, %loop : !felt.type
      constrain.eq %zeroOut, %zero : !felt.type
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto members = llvm::to_vector(structDef.getOps<MemberDefOp>());

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  const ConstraintDependencyGraph &graph = analysis.getResult(structDef);

  SourceRef sum(constrainFn.getArgument(0), {SourceRefIndex(members[0])});
  SourceRef input(constrainFn.getArgument(1));
  SourceRefSet sumDependencies = graph.getConstrainingValues(sum);
  EXPECT_TRUE(sumDependencies.contains(input));
  EXPECT_TRUE(llvm::none_of(sumDependencies, [](const SourceRef &dependency) {
    auto constant = dependency.getConstantValue();
    return succeeded(constant) && *constant == 0;
  }));

  SourceRef zeroOut(constrainFn.getArgument(0), {SourceRefIndex(members[1])});
  EXPECT_TRUE(llvm::any_of(graph.getConstrainingValues(zeroOut), [](const SourceRef &dependency) {
    auto constant = dependency.getConstantValue();
    return succeeded(constant) && *constant == 0;
  }));
}

TEST_F(SourceRefTests, DependencyStateDropsFullyOverwrittenLoopArrayZeros) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @LoopArray {
    function.def @compute(%in: !array.type<2 x !felt.type>) -> !struct.type<@LoopArray>
        attributes {function.allow_non_native_field_ops} {
      %self = struct.new : !struct.type<@LoopArray>
      %storage = llzk.nondet : !array.type<2 x !felt.type>
      %zero0 = felt.const 0
      %index0 = arith.constant 0 : index
      array.write %storage[%index0] = %zero0 : !array.type<2 x !felt.type>, !felt.type
      %zero1 = felt.const 0
      %index1 = arith.constant 1 : index
      array.write %storage[%index1] = %zero1 : !array.type<2 x !felt.type>, !felt.type
      %lower = felt.const 0
      %loop:2 = scf.while (%array = %storage, %index = %lower)
          : (!array.type<2 x !felt.type>, !felt.type)
          -> (!array.type<2 x !felt.type>, !felt.type) {
        %upper = felt.const 2
        %condition = bool.cmp lt(%index, %upper) : !felt.type, !felt.type
        scf.condition(%condition) %array, %index
            : !array.type<2 x !felt.type>, !felt.type
      } do {
      ^bb0(%array: !array.type<2 x !felt.type>, %index: !felt.type):
        %arrayIndex = cast.toindex %index : !felt.type
        %value = array.read %in[%arrayIndex] : !array.type<2 x !felt.type>, !felt.type
        array.write %array[%arrayIndex] = %value : !array.type<2 x !felt.type>, !felt.type
        %one = felt.const 1
        %next = felt.add %index, %one : !felt.type, !felt.type
        scf.yield %array, %next : !array.type<2 x !felt.type>, !felt.type
      }
      %readIndex = arith.constant 0 : index
      %value = array.read %loop#0[%readIndex] : !array.type<2 x !felt.type>, !felt.type
      function.return %self : !struct.type<@LoopArray>
    }
    function.def @constrain(
        %self: !struct.type<@LoopArray>, %in: !array.type<2 x !felt.type>) {
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  llvm::SmallVector<array::ReadArrayOp> reads;
  computeFn.walk([&](array::ReadArrayOp op) { reads.push_back(op); });
  ASSERT_EQ(reads.size(), 2U);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  SourceRefSet dependencies =
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), reads.back().getResult())
          .foldToScalar();

  EXPECT_TRUE(
      dependencies.contains(SourceRef(computeFn.getArgument(0), {SourceRefIndex(APInt(64, 0))}))
  );
  EXPECT_TRUE(llvm::none_of(dependencies, [](const SourceRef &dependency) {
    auto constant = dependency.getConstantValue();
    return succeeded(constant) && *constant == 0;
  }));
}

TEST_F(SourceRefTests, ConstraintQueriesRetainLogicalComponentPathsAlongsideInputs) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @Child {
    struct.member @out : !felt.type {llzk.pub, signal}
    function.def @compute(%in: !felt.type) -> !struct.type<@Child> {
      %self = struct.new : !struct.type<@Child>
      struct.writem %self[@out] = %in : !struct.type<@Child>, !felt.type
      function.return %self : !struct.type<@Child>
    }
    function.def @constrain(%self: !struct.type<@Child>, %in: !felt.type) {
      %out = struct.readm %self[@out] : !struct.type<@Child>, !felt.type
      %sum = felt.add %in, %in : !felt.type, !felt.type
      constrain.eq %out, %sum : !felt.type
      function.return
    }
  }

  struct.def @Parent {
    struct.member @out : !array.type<4 x !felt.type> {llzk.pub, signal}
    struct.member @child : !struct.type<@Child>
    function.def @compute(%in: !felt.type, %index: index) -> !struct.type<@Parent> {
      %self = struct.new : !struct.type<@Parent>
      function.return %self : !struct.type<@Parent>
    }
    function.def @constrain(
        %self: !struct.type<@Parent>, %in: !felt.type, %index: index
    ) {
      %out = struct.readm %self[@out] : !struct.type<@Parent>, !array.type<4 x !felt.type>
      %selected = array.read %out[%index] : !array.type<4 x !felt.type>, !felt.type
      constrain.eq %selected, %in : !felt.type
      %child = struct.readm %self[@child] : !struct.type<@Parent>, !struct.type<@Child>
      %childOut = struct.readm %child[@out] : !struct.type<@Child>, !felt.type
      %c3 = arith.constant 3 : index
      %last = array.read %out[%c3] : !array.type<4 x !felt.type>, !felt.type
      constrain.eq %last, %childOut : !felt.type
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto parent = *std::next(mod->getOps<StructDefOp>().begin());
  auto constrainFn = parent.getConstrainFuncOp();
  auto members = llvm::to_vector(parent.getOps<MemberDefOp>());
  auto child = *mod->getOps<StructDefOp>().begin();
  auto childOut = *child.getOps<MemberDefOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  analysis.ensureAnalysisRun(am);
  const ConstraintDependencyGraph &graph = analysis.getResult(parent);

  SourceRef output(
      constrainFn.getArgument(0), {SourceRefIndex(members[0]), SourceRefIndex(APInt(64, 3))}
  );
  SourceRef logicalChild(
      constrainFn.getArgument(0), {SourceRefIndex(members[1]), SourceRefIndex(childOut)}
  );
  SourceRefSet dependencies = graph.getConstrainingValues(output);
  EXPECT_TRUE(dependencies.contains(logicalChild));
  EXPECT_TRUE(dependencies.contains(SourceRef(constrainFn.getArgument(1))));
}

TEST_F(SourceRefTests, ConditionalMarkersRequireBothBooleanConstraints) {
  auto runCase = [&](bool includeTrueConstraint) {
    std::string source = R"mlir(
module attributes {llzk.lang} {
  struct.def @Conditional {
    struct.member @out : !felt.type {llzk.pub, signal}
    struct.member @helper : !felt.type {signal}
    function.def @compute(%out: !felt.type, %helper: !felt.type)
        -> !struct.type<@Conditional> {
      %self = struct.new : !struct.type<@Conditional>
      struct.writem %self[@out] = %out : !struct.type<@Conditional>, !felt.type
      struct.writem %self[@helper] = %helper : !struct.type<@Conditional>, !felt.type
      function.return %self : !struct.type<@Conditional>
    }
    function.def @constrain(
        %self: !struct.type<@Conditional>, %outArg: !felt.type, %helperArg: !felt.type
    )
        attributes {function.allow_constraint, function.allow_non_native_field_ops} {
      %out = struct.readm %self[@out] : !struct.type<@Conditional>, !felt.type
      %helper = struct.readm %self[@helper] : !struct.type<@Conditional>, !felt.type
      %zero = felt.const 0
      %one = felt.const 1
      %condition = bool.cmp eq(%helper, %zero)
      scf.if %condition {
        scf.yield
      } else {
        scf.yield
      }
      constrain.eq %out, %zero : !felt.type
)mlir";
    if (includeTrueConstraint) {
      source += "      constrain.eq %out, %one : !felt.type\n";
    }
    source += R"mlir(
      function.return
    }
  }
}
)mlir";

    auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
    EXPECT_TRUE(mod);
    if (!mod) {
      return false;
    }
    auto structDef = *mod->getOps<StructDefOp>().begin();
    auto constrainFn = structDef.getConstrainFuncOp();
    auto members = llvm::to_vector(structDef.getOps<MemberDefOp>());
    ModuleAnalysisManager mam(*mod, nullptr);
    AnalysisManager am = mam;
    ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
    analysis.ensureAnalysisRun(am);
    const ConstraintDependencyGraph &graph = analysis.getResult(structDef);
    SourceRef out(constrainFn.getArgument(0), {SourceRefIndex(members[0])});
    SourceRef helper(constrainFn.getArgument(0), {SourceRefIndex(members[1])});
    return graph.getConstrainingValues(out).contains(helper);
  };

  EXPECT_TRUE(runCase(/*includeTrueConstraint=*/true));
  EXPECT_FALSE(runCase(/*includeTrueConstraint=*/false));
}
