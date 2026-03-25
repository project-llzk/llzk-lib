//===-- Constrain.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"

#include "llzk-c/Dialect/Constrain.h"

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Constrain/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Constrain/IR/Ops.capi.test.cpp.inc"

// Implementation for `EmitEqualityOp_build_pass` test
std::unique_ptr<EmitEqualityOpBuildFuncHelper> EmitEqualityOpBuildFuncHelper::get() {
  struct Impl : public EmitEqualityOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@constrain" function as parent to avoid the following:
      // error: 'constrain.eq' op only valid within a 'function.def' with
      //        'function.allow_constraint' attribute
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructConstrain
      );
      mlir::Value val = testClass.cppGenFeltConstant(builder, location);
      return llzkConstrain_EmitEqualityOpBuild(builder, location, wrap(val), wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `EmitContainmentOp_build_pass` test
std::unique_ptr<EmitContainmentOpBuildFuncHelper> EmitContainmentOpBuildFuncHelper::get() {
  struct Impl : public EmitContainmentOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use "@constrain" function as parent to avoid the following:
      // error: 'constrain.eq' op only valid within a 'function.def' with
      //        'function.allow_constraint' attribute
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructConstrain
      );
      mlir::Value array;
      {
        mlir::OpBuilder *bldr = unwrap(builder);
        auto idxType = bldr->getIndexType();
        auto arrType = llzk::array::ArrayType::get(
            idxType, llvm::ArrayRef<mlir::Attribute> {bldr->getIntegerAttr(idxType, 0)}
        );
        array = bldr->create<llzk::array::CreateArrayOp>(unwrap(location), arrType);
      }
      return llzkConstrain_EmitContainmentOpBuild(
          builder, location, wrap(array),
          mlirOperationGetResult(testClass.createIndexOperation(), 0)
      );
    }
  };
  return std::make_unique<Impl>();
}
