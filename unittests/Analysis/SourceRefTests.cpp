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
#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Analysis/SourceRef.h"
#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Array/IR/Ops.h"
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
#include <set>

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
  SourceRefIndex range(llvm::DynamicAPInt(2), llvm::DynamicAPInt(5));
  SourceRefIndex overlappingRange(llvm::DynamicAPInt(4), llvm::DynamicAPInt(7));
  SourceRefIndex adjacentRange(llvm::DynamicAPInt(5), llvm::DynamicAPInt(8));

  EXPECT_FALSE(range.overlaps(SourceRefIndex(1)));
  EXPECT_TRUE(range.overlaps(SourceRefIndex(2)));
  EXPECT_TRUE(range.overlaps(SourceRefIndex(4)));
  EXPECT_FALSE(range.overlaps(SourceRefIndex(5)));
  EXPECT_TRUE(range.overlaps(overlappingRange));
  EXPECT_FALSE(range.overlaps(adjacentRange));
}

TEST_F(SourceRefTests, SourceRefSetHandlesConstantRefs) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  struct.def @ConstantRef {
    function.def @compute() -> !struct.type<@ConstantRef> {
      %self = struct.new : !struct.type<@ConstantRef>
      function.return %self : !struct.type<@ConstantRef>
    }

    function.def @constrain(%self: !struct.type<@ConstantRef>) {
      %zero = felt.const 0
      function.return
    }
  }
}
)mlir";

  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  felt::FeltConstantOp constant;
  mod->walk([&](felt::FeltConstantOp op) { constant = op; });
  ASSERT_TRUE(constant);

  SourceRefSet refs {SourceRef(constant)};
  EXPECT_TRUE(refs.contains(SourceRef(constant)));
}

// Dynamic dimension bounds select every nonnegative index, including values
// beyond the old unsigned 64-bit sentinel, without changing finite ranges.
TEST_F(SourceRefTests, DynamicDimensionOverlap) {
  auto dynamic = SourceRefIndex::forArrayDimension(ShapedType::kDynamic);
  auto finite = SourceRefIndex::forArrayDimension(5);
  auto empty = SourceRefIndex::forArrayDimension(0);
  SourceRefIndex huge(toDynamicAPInt("18446744073709551616"));

  EXPECT_TRUE(dynamic.hasUnboundedUpperBound());
  EXPECT_EQ(buildStringViaPrint(dynamic), "<dynamic>");
  EXPECT_TRUE(dynamic.overlaps(dynamic));
  EXPECT_TRUE(dynamic.overlaps(SourceRefIndex(0)));
  EXPECT_TRUE(dynamic.overlaps(huge));
  EXPECT_TRUE(huge.overlaps(dynamic));
  EXPECT_FALSE(dynamic.overlaps(SourceRefIndex(-1)));
  EXPECT_TRUE(dynamic.overlaps(finite));
  EXPECT_TRUE(finite.overlaps(dynamic));
  EXPECT_FALSE(dynamic.overlaps(empty));
  EXPECT_FALSE(empty.overlaps(dynamic));
  EXPECT_FALSE(finite.overlaps(SourceRefIndex(5)));

  SourceRefIndex tail(llvm::DynamicAPInt(5), llvm::DynamicAPInt(ShapedType::kDynamic));
  EXPECT_FALSE(tail.overlaps(finite));
  EXPECT_FALSE(finite.overlaps(tail));
  EXPECT_TRUE(tail.overlaps(dynamic));
}

// Exercise the three analysis paths that construct fallback access ranges:
// source-reference reads, source-reference write targets, and interval writes.
TEST_F(SourceRefTests, SymbolicArrayAccessesRetainOverlappingReferences) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  poly.template @Arrays {
    poly.param @N : index
    function.def @access(%array: !array.type<@N x !felt.type>, %i: index, %value: !felt.type)
        -> !felt.type attributes {function.allow_witness} {
      array.write %array[%i] = %value : <@N x !felt.type>, !felt.type
      %read = array.read %array[%i] : <@N x !felt.type>, !felt.type
      function.return %read : !felt.type
    }
  }
}
)mlir";
  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  array::ReadArrayOp read;
  array::WriteArrayOp write;
  mod->walk([&](array::ReadArrayOp op) { read = op; });
  mod->walk([&](array::WriteArrayOp op) { write = op; });
  ASSERT_TRUE(read);
  ASSERT_TRUE(write);

  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  ASSERT_TRUE(succeeded(llzk::dataflow::loadAndRunRequiredAnalyses(solver, *mod)));
  solver.load<SourceRefAnalysis>();
  auto *intervals =
      solver.load<IntervalDataFlowAnalysis, llvm::SMTSolverRef, const Field &, bool, bool>(
          llvm::CreateZ3Solver(), Field::getField("babybear"), false, false
      );
  ASSERT_TRUE(succeeded(solver.initializeAndRun(*mod)));

  SourceRef element(llvm::cast<BlockArgument>(read.getArrRef()), {SourceRefIndex(0)});
  auto readState = SourceRefAnalysis::getValueState(solver, read.getResult());
  ASSERT_TRUE(readState.isSingleValue());
  EXPECT_TRUE(readState.getSingleValue().overlaps(element));
  auto writeState = SourceRefAnalysis::getWriteTargetState(solver, write);
  ASSERT_TRUE(succeeded(writeState));
  ASSERT_TRUE(writeState->isSingleValue());
  EXPECT_TRUE(writeState->getSingleValue().overlaps(element));
  EXPECT_TRUE(writeState->getSingleValue().overlaps(readState.getSingleValue()));

  const auto &writes = intervals->getWriteResults();
  ASSERT_EQ(writes.size(), 1);
  EXPECT_TRUE(writes.begin()->first.overlaps(element));
  EXPECT_TRUE(writes.begin()->first.overlaps(readState.getSingleValue()));
}

TEST_F(SourceRefTests, UnboundedRangesSelectMaterializedArrayElements) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  SourceRef root(llvm::cast<OpResult>(structDef.getComputeFuncOp().getSelfValueFromCompute()));
  SourceRefLatticeValue array(llvm::ArrayRef<int64_t>({3}));
  auto dynamic = SourceRefIndex::forArrayDimension(ShapedType::kDynamic);
  EXPECT_EQ(array.write({dynamic}, SourceRefLatticeValue(root)), ChangeResult::Change);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_TRUE(array.getElemFlatIdx(i).getScalarValue().contains(root));
  }
  auto extracted = array.extract({dynamic});
  ASSERT_TRUE(succeeded(extracted));
  EXPECT_TRUE(extracted->first.getScalarValue().contains(root));
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

  SourceRef computeRoot(llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  SourceRef constrainRoot(llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()));
  SourceRef computeMember(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storage)}
  );
  SourceRef expectedMember(
      llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()), {SourceRefIndex(storage)}
  );
  SourceRef unrelated(llvm::cast<OpResult>(newPod.getResult()));

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

  // Replacing a translation after it has been consumed must rebuild its cached index.
  replacements.set(computeRoot, SourceRefLatticeValue(computeRoot));
  auto [updatedTranslation, updatedChanged] = value.translate(replacements);
  EXPECT_EQ(updatedChanged, ChangeResult::Change);
  EXPECT_TRUE(updatedTranslation.getScalarValue().contains(computeMember));
  EXPECT_FALSE(updatedTranslation.getScalarValue().contains(expectedMember));
}

TEST_F(SourceRefTests, LatticeWritesPointsSubarraysAndRanges) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  SourceRef computeRoot(llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  SourceRef constrainRoot(llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()));

  SourceRefLatticeValue matrix(llvm::ArrayRef<int64_t>({2, 2}));
  EXPECT_EQ(
      matrix.write(
          {SourceRefIndex(llvm::DynamicAPInt(0)), SourceRefIndex(llvm::DynamicAPInt(1))},
          SourceRefLatticeValue(computeRoot)
      ),
      ChangeResult::Change
  );
  auto point = matrix.extract(
      {SourceRefIndex(llvm::DynamicAPInt(0)), SourceRefIndex(llvm::DynamicAPInt(1))}
  );
  ASSERT_TRUE(succeeded(point));
  EXPECT_EQ(point->first.getSingleValue(), computeRoot);

  SourceRefLatticeValue row(llvm::ArrayRef<int64_t>({2}));
  EXPECT_EQ(
      row.getElemFlatIdx(0).setValue(SourceRefLatticeValue(constrainRoot)), ChangeResult::Change
  );
  EXPECT_EQ(
      row.getElemFlatIdx(1).setValue(SourceRefLatticeValue(computeRoot)), ChangeResult::Change
  );
  EXPECT_EQ(matrix.write({SourceRefIndex(llvm::DynamicAPInt(1))}, row), ChangeResult::Change);
  auto writtenRow = matrix.extract({SourceRefIndex(llvm::DynamicAPInt(1))});
  ASSERT_TRUE(succeeded(writtenRow));
  ASSERT_TRUE(writtenRow->first.isArray());
  EXPECT_EQ(writtenRow->first.getElemFlatIdx(0).getSingleValue(), constrainRoot);
  EXPECT_EQ(writtenRow->first.getElemFlatIdx(1).getSingleValue(), computeRoot);

  SourceRefLatticeValue vector(llvm::ArrayRef<int64_t>({3}));
  EXPECT_EQ(
      vector.write(
          {SourceRefIndex(llvm::DynamicAPInt(1), llvm::DynamicAPInt(3))},
          SourceRefLatticeValue(constrainRoot)
      ),
      ChangeResult::Change
  );
  for (int64_t index = 1; index < 3; ++index) {
    auto ranged = vector.extract({SourceRefIndex(llvm::DynamicAPInt(index))});
    ASSERT_TRUE(succeeded(ranged));
    EXPECT_TRUE(ranged->first.getScalarValue().contains(constrainRoot));
  }

  SourceRefLatticeValue tensor(llvm::ArrayRef<int64_t>({2, 2, 3}));
  SourceRefLatticeValue matrixSlice(llvm::ArrayRef<int64_t>({2, 3}));
  EXPECT_EQ(
      matrixSlice.getElemFlatIdx(0).setValue(SourceRefLatticeValue(computeRoot)),
      ChangeResult::Change
  );
  EXPECT_EQ(
      tensor.write({SourceRefIndex(llvm::DynamicAPInt(1))}, matrixSlice), ChangeResult::Change
  );

  SourceRefLatticeValue transposedSlice(llvm::ArrayRef<int64_t>({3, 2}));
  EXPECT_DEATH(
      (void)tensor.write({SourceRefIndex(llvm::DynamicAPInt(0))}, transposedSlice),
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
  auto constrainSelf = llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  SourceRef computeMember(llvm::cast<OpResult>(returnedSelf), {SourceRefIndex(storage)});
  SourceRef temporaryMember(llvm::cast<OpResult>(temporary.getResult()), {SourceRefIndex(storage)});
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
  auto constrainSelf = llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  auto *successor = new Block();
  constrainFn.getBody().push_back(successor);
  auto successorArg = successor->addArgument(constrainSelf.getType(), loc);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(successor);
  llzk::function::ReturnOp::create(builder, loc);

  SourceRef computeMember(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storage)}
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
  SourceRef root(llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  SourceRef memberRef(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()), {SourceRefIndex(storage)}
  );
  SourceRef podRef(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()),
      {SourceRefIndex(StringAttr::get(&ctx, "storage"))}
  );

  EXPECT_FALSE(memberRef.isValidPrefix(podRef));
  EXPECT_FALSE(podRef.isValidPrefix(memberRef));
  EXPECT_FALSE(memberRef.overlaps(podRef));
  EXPECT_FALSE(podRef.overlaps(memberRef));
  EXPECT_TRUE(memberRef.isValidPrefix(root));

  SourceRef arbitraryPodRef(
      llvm::cast<OpResult>(newPod.getResult()), {SourceRefIndex(StringAttr::get(&ctx, "storage"))}
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
  computeFn.walk([&](pod::NewPodOp op) { pods.push_back(op); });
  computeFn.walk([&](pod::ReadPodOp op) { reads.push_back(op); });
  ASSERT_EQ(pods.size(), 5U);
  ASSERT_EQ(reads.size(), 7U);
  pod::ReadPodOp constrainRead;
  constrainFn.walk([&](pod::ReadPodOp op) { constrainRead = op; });
  ASSERT_TRUE(constrainRead);

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  ASSERT_TRUE(succeeded(analysis.ensureAnalysisRun(am)));
  DataFlowSolver &solver = analysis.getSolver();

  SourceRef constrainStorageValue(
      mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain()),
      {SourceRefIndex(storageMember), SourceRefIndex(StringAttr::get(&ctx, "value"))}
  );
  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(solver, constrainRead.getResult()).foldToScalar(),
      SourceRefSet({constrainStorageValue})
  );

  auto valueName = StringAttr::get(&ctx, "value");
  auto missingName = StringAttr::get(&ctx, "missing");
  SourceRef storageValue(mlir::cast<OpResult>(pods[0].getResult()), {SourceRefIndex(valueName)});
  SourceRef storageMissing(
      mlir::cast<OpResult>(pods[0].getResult()), {SourceRefIndex(missingName)}
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

  auto maybeDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[3].getResult()).foldToScalar();
  EXPECT_TRUE(maybeDependencies.contains(SourceRef(computeFn.getArgument(0))));
  EXPECT_TRUE(maybeDependencies.contains(SourceRef(computeFn.getArgument(1))));

  auto cycleDependencies =
      SourceRefAnalysis::getDependencyState(solver, reads[6].getResult()).foldToScalar();
  EXPECT_FALSE(cycleDependencies.empty());
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
  ASSERT_TRUE(succeeded(analysis.ensureAnalysisRun(am)));

  SourceRef parentInput(parentConstrain.getArgument(1));
  SourceRef zeroRef(zero);
  EXPECT_TRUE(analysis.getResult(structs[1]).getConstrainingValues(parentInput).contains(zeroRef));
}

/// Returned arrays retain physical addresses independently of initialized dependencies.
TEST_F(SourceRefTests, ReturnedArraysKeepAddressesSeparateFromDependencies) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  function.def @make(%value: !felt.type) -> !array.type<1 x !felt.type> {
    %array = array.new %value : !array.type<1 x !felt.type>
    function.return %array : !array.type<1 x !felt.type>
  }
  struct.def @ReturnedStorage {
    function.def @compute(%initial: !felt.type, %replacement: !felt.type)
        -> !struct.type<@ReturnedStorage> {
      %self = struct.new : !struct.type<@ReturnedStorage>
      function.return %self : !struct.type<@ReturnedStorage>
    }
    function.def @constrain(%self: !struct.type<@ReturnedStorage>,
        %initial: !felt.type, %replacement: !felt.type) {
      %first = function.call @make(%initial) : (!felt.type) -> !array.type<1 x !felt.type>
      %second = function.call @make(%initial) : (!felt.type) -> !array.type<1 x !felt.type>
      %c0 = arith.constant 0 : index
      array.write %first[%c0] = %replacement : !array.type<1 x !felt.type>, !felt.type
      %read = array.read %second[%c0] : !array.type<1 x !felt.type>, !felt.type
      function.return
    }
  }
}
)mlir";
  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto calls = llvm::to_vector(constrainFn.getOps<function::CallOp>());
  ASSERT_EQ(calls.size(), 2U);
  auto write = *constrainFn.getOps<array::WriteArrayOp>().begin();
  auto read = *constrainFn.getOps<array::ReadArrayOp>().begin();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  ASSERT_TRUE(succeeded(analysis.ensureAnalysisRun(am)));
  DataFlowSolver &solver = analysis.getSolver();

  SourceRef firstAddress(mlir::cast<OpResult>(calls[0].getResult(0)), {SourceRefIndex(0)});
  SourceRef secondAddress(mlir::cast<OpResult>(calls[1].getResult(0)), {SourceRefIndex(0)});
  EXPECT_NE(firstAddress, secondAddress);
  llvm::SmallVector<SourceRef> expectedAddresses {firstAddress, secondAddress};
  for (auto [call, expected] : llvm::zip(calls, expectedAddresses)) {
    auto raw = SourceRefAnalysis::getValueState(solver, call.getResult(0));
    ASSERT_TRUE(raw.isArray());
    ASSERT_EQ(raw.getArrayShape(), std::vector<int64_t>({1}));
    EXPECT_EQ(raw.foldToScalar(), SourceRefSet({expected}));
    EXPECT_EQ(
        SourceRefAnalysis::getDependencyState(solver, call.getResult(0)).foldToScalar(),
        SourceRefSet({SourceRef(constrainFn.getArgument(1))})
    );
  }
  auto target = SourceRefAnalysis::getWriteTargetState(solver, write);
  ASSERT_TRUE(succeeded(target));
  ASSERT_TRUE(target->isSingleValue());
  EXPECT_EQ(target->getSingleValue(), firstAddress);
  EXPECT_EQ(
      SourceRefAnalysis::getValueState(solver, read.getResult()).foldToScalar(),
      SourceRefSet({secondAddress})
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
  ASSERT_TRUE(succeeded(analysis.ensureAnalysisRun(am)));

  EXPECT_EQ(
      SourceRefAnalysis::getDependencyState(analysis.getSolver(), reads[1].getResult())
          .foldToScalar(),
      SourceRefSet({SourceRef(computeFn.getArgument(0))})
  );
}

// Member writes require witness code, so inspect the read through the analysis API rather
// than adding a constraint to compute. Copying the slot freezes its handle, not its members.
TEST_F(SourceRefTests, SymbolicCopyCapturesHandleButObservesLaterMemberWrite) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  poly.template @Cells {
    poly.param @N : index
    struct.def @Cell {
      struct.member @value : !pod.type<[@item: !felt.type]>
      function.def @compute(
          %source: !array.type<@N x !struct.type<@Cells::@Cell<[@N]>>>,
          %initial: !felt.type, %later: !felt.type
      ) -> !struct.type<@Cells::@Cell<[@N]>> {
        %self = struct.new : !struct.type<@Cells::@Cell<[@N]>>
        %other = struct.new : !struct.type<@Cells::@Cell<[@N]>>
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %initialPod = pod.new { @item = %initial } : !pod.type<[@item: !felt.type]>
        struct.writem %self[@value] = %initialPod
            : !struct.type<@Cells::@Cell<[@N]>>, !pod.type<[@item: !felt.type]>
        array.write %source[%c0] = %other
            : <@N x !struct.type<@Cells::@Cell<[@N]>>>, !struct.type<@Cells::@Cell<[@N]>>
        array.write %source[%c1] = %self
            : <@N x !struct.type<@Cells::@Cell<[@N]>>>, !struct.type<@Cells::@Cell<[@N]>>
        %holder = pod.new { @saved = %source }
            : !pod.type<[@saved: !array.type<@N x !struct.type<@Cells::@Cell<[@N]>>>]>
        %copy = pod.read %holder[@saved]
            : !pod.type<[@saved: !array.type<@N x !struct.type<@Cells::@Cell<[@N]>>>]>,
              !array.type<@N x !struct.type<@Cells::@Cell<[@N]>>>
        array.write %source[%c1] = %other
            : <@N x !struct.type<@Cells::@Cell<[@N]>>>, !struct.type<@Cells::@Cell<[@N]>>
        %laterPod = pod.new { @item = %later } : !pod.type<[@item: !felt.type]>
        struct.writem %self[@value] = %laterPod
            : !struct.type<@Cells::@Cell<[@N]>>, !pod.type<[@item: !felt.type]>
        %selected = array.read %copy[%c1]
            : <@N x !struct.type<@Cells::@Cell<[@N]>>>, !struct.type<@Cells::@Cell<[@N]>>
        %member = struct.readm %selected[@value]
            : !struct.type<@Cells::@Cell<[@N]>>, !pod.type<[@item: !felt.type]>
        %observed = pod.read %member[@item] : !pod.type<[@item: !felt.type]>, !felt.type
        function.return %self : !struct.type<@Cells::@Cell<[@N]>>
      }
      function.def @constrain(
          %self: !struct.type<@Cells::@Cell<[@N]>>,
          %source: !array.type<@N x !struct.type<@Cells::@Cell<[@N]>>>,
          %initial: !felt.type, %later: !felt.type
      ) { function.return }
    }
  }
}
)mlir";
  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  pod::ReadPodOp read;
  mod->walk([&](pod::ReadPodOp op) {
    if (llvm::isa<felt::FeltType>(op.getResult().getType())) {
      read = op;
    }
  });
  ASSERT_TRUE(read);
  auto compute = read->getParentOfType<function::FuncDefOp>();

  ModuleAnalysisManager mam(*mod, nullptr);
  AnalysisManager am = mam;
  ConstraintDependencyGraphModuleAnalysis analysis(mod->getOperation());
  ASSERT_TRUE(succeeded(analysis.ensureAnalysisRun(am)));
  auto dependencies = SourceRefAnalysis::getDependencyState(analysis.getSolver(), read.getResult());
  EXPECT_EQ(dependencies.foldToScalar(), SourceRefSet({SourceRef(compute.getArgument(2))}))
      << buildStringViaPrint(dependencies);
}

TEST_F(SourceRefTests, ComputeSelfRebasesToConstrainSelfWithoutChangingPath) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto computeFn = structDef.getComputeFuncOp();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  auto valueName = StringAttr::get(&ctx, "value");

  SourceRef computeSelf(llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()));
  auto constrainSelfArg = llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());
  SourceRef constrainSelf(constrainSelfArg);
  SourceRef computeValue(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()),
      {SourceRefIndex(storage), SourceRefIndex(valueName)}
  );
  SourceRef expectedConstrainValue(
      constrainSelfArg, {SourceRefIndex(storage), SourceRefIndex(valueName)}
  );

  auto suffix = computeValue.getSuffix(computeSelf);
  ASSERT_TRUE(succeeded(suffix));
  EXPECT_EQ(SourceRef::appendSuffix(constrainSelf, *suffix), expectedConstrainValue);

  auto translated = computeValue.translate(computeSelf, constrainSelf);
  ASSERT_TRUE(succeeded(translated));
  EXPECT_EQ(*translated, expectedConstrainValue);

  SourceRef mismatchedComputeValue(
      llvm::cast<OpResult>(computeFn.getSelfValueFromCompute()),
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
  auto constrainSelf = llvm::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  auto *successor = new Block();
  constrainFn.getBody().push_back(successor);
  auto successorArg = successor->addArgument(constrainSelf.getType(), loc);
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(successor);
  llzk::function::ReturnOp::create(builder, loc);

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

TEST_F(SourceRefTests, ConstantFeltValueUsesFieldRepresentative) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto attr = felt::FeltConstAttr::get(&ctx, DynamicAPInt(2013265928), "babybear");
  auto constant = felt::FeltConstantOp::create(builder, loc, attr);
  SourceRef ref(constant);
  auto value = ref.getConstantFeltValue();
  ASSERT_TRUE(succeeded(value));
  EXPECT_EQ(*value, DynamicAPInt(7));
}

/// Fieldless constants have no numeric field representative until a field is supplied.
/// Their equality, hashing, ordering, and printing must still use the raw value:
/// 2013265928 reduces to 7 in BabyBear, but remains distinct from a fieldless 7.
TEST_F(SourceRefTests, FieldlessConstantsSeparateIdentityFromEvaluation) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto attr = felt::FeltConstAttr::get(&ctx, DynamicAPInt(2013265928));
  auto first = felt::FeltConstantOp::create(builder, loc, attr);
  auto second = felt::FeltConstantOp::create(builder, loc, attr);
  auto seven =
      felt::FeltConstantOp::create(builder, loc, felt::FeltConstAttr::get(&ctx, DynamicAPInt(7)));
  EXPECT_FALSE(first.getValue().getReducedValue());
  SourceRef a(first), b(second), c(seven);
  EXPECT_TRUE(failed(a.getConstantFeltValue()));
  EXPECT_TRUE(failed(a.getConstantValue()));
  auto value = a.getConstantValue(Field::getField("babybear"));
  ASSERT_TRUE(succeeded(value));
  EXPECT_EQ(*value, DynamicAPInt(7));
  EXPECT_EQ(a, b);
  EXPECT_EQ(SourceRef::Hash {}(a), SourceRef::Hash {}(b));
  EXPECT_NE(a, c);
  EXPECT_EQ(a <=> b, std::strong_ordering::equal);
  EXPECT_NE(a <=> c, std::strong_ordering::equal);
  EXPECT_EQ(buildStringViaPrint(a), "<felt.const: 2013265928>");
}

/// Selecting an analysis field preserves the constant's declared-field representative.
TEST_F(SourceRefTests, SuppliedFieldPreservesDeclaredConstantField) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto attr = felt::FeltConstAttr::get(&ctx, DynamicAPInt(2013265928), "babybear");
  auto constant = felt::FeltConstantOp::create(builder, loc, attr);
  SourceRef ref(constant);
  auto declaredValue = ref.getConstantValue();
  ASSERT_TRUE(succeeded(declaredValue));
  EXPECT_EQ(*declaredValue, DynamicAPInt(7));
  auto analysisValue = ref.getConstantValue(Field::getField("koalabear"));
  ASSERT_TRUE(succeeded(analysisValue));
  EXPECT_EQ(*analysisValue, DynamicAPInt(7));
}

/// Signed literals use their declared representative before reduction into a smaller field.
TEST_F(SourceRefTests, SuppliedFieldReducesDeclaredSignedRepresentative) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto attr = felt::FeltConstAttr::get(&ctx, DynamicAPInt(-1), "koalabear");
  auto constant = felt::FeltConstantOp::create(builder, loc, attr);
  SourceRef ref(constant);
  auto declaredValue = ref.getConstantValue();
  ASSERT_TRUE(succeeded(declaredValue));
  EXPECT_EQ(*declaredValue, DynamicAPInt(2130706432));
  auto analysisValue = ref.getConstantValue(Field::getField("babybear"));
  ASSERT_TRUE(succeeded(analysisValue));
  EXPECT_EQ(*analysisValue, DynamicAPInt(117440511));
}

// Numeric source-reference keys must hash by value, independent of whether a
// DynamicAPInt was constructed from a native integer or a 256-bit APInt. Cover
// both individual indices and ranges so equal keys work in hash containers.
TEST_F(SourceRefTests, NumericKeyHashIgnoresIntegerRepresentation) {
  DynamicAPInt narrow(7), wide(llvm::APInt(256, 7));
  SourceRefIndex a(narrow), b(wide);
  EXPECT_EQ(a, b);
  EXPECT_EQ(SourceRefIndex::Hash {}(a), SourceRefIndex::Hash {}(b));
  SourceRefIndex x(std::pair {narrow, narrow + 3});
  SourceRefIndex y(std::pair {wide, wide + 3});
  EXPECT_EQ(x, y);
  EXPECT_EQ(SourceRefIndex::Hash {}(x), SourceRefIndex::Hash {}(y));
}

/// Ordered containers retain equal numeric representatives with distinct felt types.
TEST_F(SourceRefTests, ConstantOrderingDistinguishesFeltTypes) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto makeRef = [&](int64_t value, felt::FeltType type) {
    auto attr = felt::FeltConstAttr::get(&ctx, DynamicAPInt(value), type);
    return SourceRef(felt::FeltConstantOp::create(builder, loc, attr));
  };
  auto babybear = felt::FeltType::get(&ctx, "babybear");
  auto a = makeRef(2013265928, babybear);
  auto b = makeRef(7, felt::FeltType::get(&ctx, "koalabear"));
  auto c = makeRef(7, felt::FeltType::get(&ctx));
  auto canonical = makeRef(7, babybear);
  for (const auto &other : {b, c}) {
    EXPECT_NE(a, other);
    EXPECT_NE(a <=> other, std::strong_ordering::equal);
    EXPECT_NE(a < other, other < a);
  }
  EXPECT_NE(b, c);
  EXPECT_NE(b <=> c, std::strong_ordering::equal);
  EXPECT_EQ(a, canonical);
  EXPECT_EQ(a <=> canonical, std::strong_ordering::equal);
  std::set<SourceRef> refs {a, b, c, canonical};
  EXPECT_EQ(refs.size(), 3U);
}

// Congruent signed literals share semantic identity once their field is known.
TEST_F(SourceRefTests, SignedRepresentativesShareConstantIdentity) {
  Block block;
  OpBuilder builder(&ctx);
  builder.setInsertionPointToEnd(&block);
  auto makeRef = [&](int64_t value) {
    auto attr = felt::FeltConstAttr::get(&ctx, DynamicAPInt(value), "babybear");
    return SourceRef(felt::FeltConstantOp::create(builder, loc, attr));
  };
  auto negative = makeRef(-1);
  auto canonical = makeRef(2013265920);
  EXPECT_EQ(negative, canonical);
  EXPECT_EQ(SourceRef::Hash {}(negative), SourceRef::Hash {}(canonical));
  EXPECT_EQ(negative <=> canonical, std::strong_ordering::equal);
  EXPECT_EQ(buildStringViaPrint(negative), "<felt.const: 2013265920>");
}

/// Context is part of identity everywhere references are compared or used as storage prefixes.
TEST_F(SourceRefTests, CallContextsPreserveIdentityAndPathOperations) {
  static constexpr auto source = R"mlir(
module attributes {llzk.lang} {
  function.def @make() -> !pod.type<[@left: !felt.type, @right: !felt.type]> {
    %pod = pod.new : !pod.type<[@left: !felt.type, @right: !felt.type]>
    function.return %pod : !pod.type<[@left: !felt.type, @right: !felt.type]>
  }
  function.def @use() {
    %first = function.call @make() : () -> !pod.type<[@left: !felt.type, @right: !felt.type]>
    %second = function.call @make() : () -> !pod.type<[@left: !felt.type, @right: !felt.type]>
    %zero = felt.const 0
    function.return
  }
}
)mlir";
  auto mod = parseSourceString<ModuleOp>(source, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto functions = llvm::to_vector(mod->getOps<function::FuncDefOp>());
  auto allocation = *functions[0].getOps<pod::NewPodOp>().begin();
  auto calls = llvm::to_vector(functions[1].getOps<function::CallOp>());
  auto zero = *functions[1].getOps<felt::FeltConstantOp>().begin();
  SourceRef root(llvm::cast<OpResult>(allocation.getResult()));
  SourceRefIndex leftIndex(StringAttr::get(&ctx, "left"));
  SourceRefIndex rightIndex(StringAttr::get(&ctx, "right"));
  SourceRef left = *root.createChild(leftIndex);
  SourceRef firstRoot = root.inCallContext(calls[0]);
  SourceRef secondRoot = root.inCallContext(calls[1]);
  SourceRef firstLeft = left.inCallContext(calls[0]);
  SourceRef secondLeft = left.inCallContext(calls[1]);
  SourceRef firstRight = *firstRoot.createChild(rightIndex);

  EXPECT_EQ(firstLeft, *firstRoot.createChild(leftIndex));
  EXPECT_NE(firstLeft, secondLeft);
  EXPECT_EQ(*firstLeft.getParentPrefix(), firstRoot);
  EXPECT_EQ(firstLeft.getType(), left.getType());
  EXPECT_EQ(*firstLeft.getRoot(), *left.getRoot());
  EXPECT_TRUE(firstLeft.isValidPrefix(firstRoot));
  EXPECT_FALSE(firstLeft.isValidPrefix(secondRoot));
  EXPECT_FALSE(firstLeft.isValidPrefix(root));
  EXPECT_TRUE(firstLeft.overlaps(firstLeft));
  EXPECT_FALSE(firstLeft.overlaps(secondLeft));
  EXPECT_FALSE(firstLeft.overlaps(firstRight));
  EXPECT_EQ(*firstLeft.translate(firstRoot, secondRoot), secondLeft);
  EXPECT_EQ(SourceRef(zero).inCallContext(calls[0]), SourceRef(zero));

  SourceRefSet hashed {left, firstLeft, secondLeft, firstRight, left.inCallContext(calls[0])};
  std::set<SourceRef> ordered(hashed.begin(), hashed.end());
  EXPECT_EQ(hashed.size(), 4U);
  EXPECT_EQ(ordered.size(), hashed.size());
  EXPECT_EQ(SourceRef::Hash {}(firstLeft), SourceRef::Hash {}(left.inCallContext(calls[0])));
  // A recursive revisit reaches the same finite identity instead of growing a call string.
  EXPECT_EQ(firstLeft.inCallContext(calls[1]).inCallContext(calls[0]), firstLeft);
}

TEST_F(SourceRefTests, CapturedComponentsHaveStableValueIdentities) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  Type cellType = structDef.getConstrainFuncOp().getArgument(0).getType();
  Block block;
  SourceRef root(block.addArgument(array::ArrayType::get(cellType, {1}), UnknownLoc::get(&ctx)));
  auto slot = root.createChild(SourceRefIndex(0));
  ASSERT_TRUE(succeeded(slot));
  SourceRef captured = slot->captureComponent();
  EXPECT_NE(captured, *slot);
  EXPECT_EQ(captured.captureComponent(), captured);
  EXPECT_EQ(captured.withoutComponentCapture(), *slot);
  EXPECT_EQ((llvm::DenseSet<SourceRef> {*slot, captured, captured.captureComponent()}).size(), 2);
  EXPECT_EQ((std::set<SourceRef> {*slot, captured, captured.captureComponent()}).size(), 2);

  auto member = captured.createChild(SourceRefIndex(*structDef.getOps<MemberDefOp>().begin()));
  ASSERT_TRUE(succeeded(member));
  EXPECT_TRUE(member->protectsCapturedComponent(*slot));
  EXPECT_TRUE(member->protectsCapturedComponent(root));
  auto rangedSlot = root.createChild(SourceRefIndex(llvm::DynamicAPInt(0), llvm::DynamicAPInt(2)));
  auto otherSlot = root.createChild(SourceRefIndex(1));
  ASSERT_TRUE(succeeded(rangedSlot));
  ASSERT_TRUE(succeeded(otherSlot));
  EXPECT_TRUE(member->protectsCapturedComponent(*rangedSlot));
  EXPECT_FALSE(member->protectsCapturedComponent(*otherSlot));
  EXPECT_FALSE(member->protectsCapturedComponent(*member));
  EXPECT_EQ(*member->getParentPrefix(), captured);
  EXPECT_EQ(*captured.getParentPrefix(), root);
}

TEST_F(SourceRefTests, CapturedComponentBoundariesSurviveTranslation) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto self = structDef.getConstrainFuncOp().getArgument(0);
  Block block;
  SourceRef root(
      block.addArgument(array::ArrayType::get(self.getType(), {1}), UnknownLoc::get(&ctx))
  );
  SourceRef matrix(
      block.addArgument(array::ArrayType::get(self.getType(), {2, 1}), UnknownLoc::get(&ctx))
  );
  auto slot = root.createChild(SourceRefIndex(0));
  auto row = matrix.createChild(SourceRefIndex(1));
  ASSERT_TRUE(succeeded(slot));
  ASSERT_TRUE(succeeded(row));
  SourceRefIndex memberIndex(*structDef.getOps<MemberDefOp>().begin());
  auto member = slot->captureComponent().createChild(memberIndex);
  ASSERT_TRUE(succeeded(member));
  auto translated = member->translate(root, *row);
  ASSERT_TRUE(succeeded(translated));
  auto translatedSlot = row->createChild(SourceRefIndex(0));
  ASSERT_TRUE(succeeded(translatedSlot));
  EXPECT_TRUE(translated->protectsCapturedComponent(*translatedSlot));
  EXPECT_EQ(*translated->getParentPrefix(), translatedSlot->captureComponent());
  EXPECT_EQ(*translated->getParentPrefix()->getParentPrefix(), *row);

  TranslationMap aliases {{*slot, SourceRefLatticeValue(SourceRef(self))}};
  auto [preserved, unchanged] =
      SourceRefLatticeValue(*member).replacePrefixes(aliases, /*preserveCaptures=*/true);
  EXPECT_EQ(unchanged, ChangeResult::NoChange);
  EXPECT_EQ(preserved.getSingleValue(), *member);
  auto [rebound, changed] = SourceRefLatticeValue(*member).replacePrefixes(aliases);
  EXPECT_EQ(changed, ChangeResult::Change);
  EXPECT_EQ(rebound.getSingleValue(), *SourceRef(self).createChild(memberIndex));
}
