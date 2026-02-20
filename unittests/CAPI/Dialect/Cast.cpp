//===-- Cast.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Cast.h"

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Cast/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Cast/IR/Ops.capi.test.cpp.inc"

// Implementation for `IntToFeltOp_build_pass` test
std::unique_ptr<IntToFeltOpBuildFuncHelper> IntToFeltOpBuildFuncHelper::get() {
  struct Impl : public IntToFeltOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::Operation *> forceCleanup;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirOperation op = testClass.createIndexOperation();
      this->forceCleanup = unwrap(op);
      return llzkCast_IntToFeltOpBuildWithType(
          builder, location, wrap(testClass.cppGetFeltType(builder)), mlirOperationGetResult(op, 0)
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `FeltToIndexOp_build_pass` test
std::unique_ptr<FeltToIndexOpBuildFuncHelper> FeltToIndexOpBuildFuncHelper::get() {
  struct Impl : public FeltToIndexOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::Operation *> forceCleanup;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      mlir::Value val = testClass.cppGenFeltConstant(builder, location);
      this->forceCleanup = val.getDefiningOp();
      return llzkCast_FeltToIndexOpBuild(builder, location, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}
