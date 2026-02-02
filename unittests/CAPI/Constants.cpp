//===-- Constants.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Constants.h"

#include <gtest/gtest.h>

TEST(LLZKConstants, HaveCorrectValues) {
  EXPECT_EQ(strcmp(LLZK_FUNC_NAME_COMPUTE, "compute"), 0);
  EXPECT_EQ(strcmp(LLZK_FUNC_NAME_CONSTRAIN, "constrain"), 0);
  EXPECT_EQ(strcmp(LLZK_LANG_ATTR_NAME, "veridise.lang"), 0);
  EXPECT_EQ(strcmp(LLZK_MAIN_ATTR_NAME, "llzk.main"), 0);
}
