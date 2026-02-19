//===-- String.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/String.h"

#include <mlir-c/BuiltinTypes.h>

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_string) {
  (void)mlirGetDialectHandle__llzk__string__();
}

TEST_F(CAPITest, llzk_string_type_get) {
  auto type = llzkString_StringTypeGet(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_string_type_pass) {
  auto type = llzkString_StringTypeGet(context);
  EXPECT_TRUE(llzkTypeIsA_String_StringType(type));
}

TEST_F(CAPITest, llzk_type_is_a_string_type_fail) {
  auto type = mlirIndexTypeGet(context);
  EXPECT_TRUE(!llzkTypeIsA_String_StringType(type));
}
