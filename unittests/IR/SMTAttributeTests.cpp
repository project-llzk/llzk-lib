//===-- SMTAttributeTests.cpp - Unit tests for SMT attributes ---*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "../LLZKTestBase.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"

#include <llvm/ADT/APInt.h>

class SMTAttributeTests : public LLZKTest {};

TEST_F(SMTAttributeTests, NumericAPIntStorageReusesEqualValuesAcrossWidths) {
  auto expectStorageReuse = [&](const llvm::APInt &narrow, const llvm::APInt &wide) {
    ASSERT_TRUE(llvm::APInt::isSameValue(narrow, wide));
    EXPECT_EQ(
        llvm::hash_combine(llzk::APIntValue(narrow)), llvm::hash_combine(llzk::APIntValue(wide))
    );

    llzk::felt::FeltConstAttr narrowAttr = llzk::felt::FeltConstAttr::get(&ctx, narrow);
    llzk::felt::FeltConstAttr wideAttr = llzk::felt::FeltConstAttr::get(&ctx, wide);
    EXPECT_EQ(narrowAttr, wideAttr);
  };

  expectStorageReuse(llvm::APInt(1, 0), llvm::APInt(128, 0));

  llvm::APInt multiword = llvm::APInt::getOneBitSet(65, 64) | llvm::APInt(65, 7);
  expectStorageReuse(multiword, multiword.zext(129));
}
