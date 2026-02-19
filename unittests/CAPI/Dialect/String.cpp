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

// Include the auto-generated tests
#include "llzk/Dialect/String/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/String/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/String/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_string_type_get) {
  auto type = llzkString_StringTypeGet(context);
  EXPECT_NE(type.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_string_type_pass) {
  auto type = llzkString_StringTypeGet(context);
  EXPECT_TRUE(llzkTypeIsA_String_StringType(type));
}

// Implementation for `LitStringOp_build_pass` test
std::unique_ptr<LitStringOpBuildFuncHelper> LitStringOpBuildFuncHelper::get() {
  struct Impl : public LitStringOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirIdentifier strVal =
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString("test"));
      return llzkString_LitStringOpBuild(
          builder, location, llzkString_StringTypeGet(testClass.context), strVal
      );
    }
  };
  return std::make_unique<Impl>();
}
