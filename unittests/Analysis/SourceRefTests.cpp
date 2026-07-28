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

#include "llzk/Analysis/SourceRef.h"
#include "llzk/Analysis/SourceRefLattice.h"
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

TEST_F(SourceRefTests, ConstrainSelfPrintsAsSelf) {
  auto mod = parseSourceString<ModuleOp>(kModule, ParserConfig(&ctx));
  ASSERT_TRUE(mod);
  auto structDef = *mod->getOps<StructDefOp>().begin();
  auto constrainFn = structDef.getConstrainFuncOp();
  auto storage = *structDef.getOps<MemberDefOp>().begin();
  auto constrainSelf = mlir::cast<BlockArgument>(constrainFn.getSelfValueFromConstrain());

  EXPECT_EQ(buildStringViaPrint(SourceRef(constrainSelf)), "%self");
  EXPECT_EQ(
      buildStringViaPrint(SourceRef(constrainSelf, {SourceRefIndex(storage)})), "%self.storage"
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
