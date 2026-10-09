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

#include "llzk/Analysis/IntervalAnalysis.h"
#include "llzk/Analysis/SourceRef.h"
#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Array/IR/Ops.h"
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
