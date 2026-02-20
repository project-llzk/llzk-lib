//===-- Boolean.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Bool.h"

#include "../CAPITestBase.h"

// Include necessary generated CAPI
#include "llzk/Dialect/Bool/IR/Enums.capi.cpp.inc"

// Include the auto-generated tests
#include "llzk/Dialect/Bool/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Bool/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Bool/IR/Enums.capi.test.cpp.inc"
#include "llzk/Dialect/Bool/IR/Ops.capi.test.cpp.inc"

class CmpAttrTest : public CAPITest,
                    public testing::WithParamInterface<LlzkBoolFeltCmpPredicate> {};

TEST_P(CmpAttrTest, llzk_felt_cmp_predicate_attr_get) {
  auto attr = llzkBool_FeltCmpPredicateAttrGet(context, GetParam());
  EXPECT_NE(attr.ptr, (void *)NULL);
}

INSTANTIATE_TEST_SUITE_P(
    AllLlzkBoolFeltCmpPredicateValues, CmpAttrTest,
    testing::Values(
        LlzkBoolFeltCmpPredicate_EQ, LlzkBoolFeltCmpPredicate_LE, LlzkBoolFeltCmpPredicate_LT,
        LlzkBoolFeltCmpPredicate_GE, LlzkBoolFeltCmpPredicate_GT, LlzkBoolFeltCmpPredicate_NE
    )
);

TEST_F(CAPITest, llzk_attribute_is_a_felt_cmp_predicate_attr_pass) {
  auto attr = llzkBool_FeltCmpPredicateAttrGet(context, LlzkBoolFeltCmpPredicate_EQ);
  EXPECT_TRUE(llzkAttributeIsA_Bool_FeltCmpPredicateAttr(attr));
}

// Implementation for `CmpOp_build_pass` test
std::unique_ptr<CmpOpBuildFuncHelper> CmpOpBuildFuncHelper::get() {
  struct Impl : public CmpOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Attribute cppAttr = unwrap(builder)->getAttr<llzk::boolean::FeltCmpPredicateAttr>(
          llzk::boolean::FeltCmpPredicate::EQ
      );
      mlir::Value cppValue = CAPITest::cppGenFeltConstant(builder, location);
      return llzkBool_CmpOpBuild(builder, location, wrap(cppValue), wrap(cppValue), wrap(cppAttr));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `AssertOp_build_pass` test
std::unique_ptr<AssertOpBuildFuncHelper> AssertOpBuildFuncHelper::get() {
  struct Impl : public AssertOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      mlir::Value cppValue = CAPITest::cppGenBoolConstant(builder, location);
      return llzkBool_AssertOpBuild(
          builder, location, wrap(cppValue), MlirIdentifier {.ptr = NULL}
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `OrBoolOp_build_pass` test
std::unique_ptr<OrBoolOpBuildFuncHelper> OrBoolOpBuildFuncHelper::get() {
  struct Impl : public OrBoolOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      mlir::Value cppValue = CAPITest::cppGenBoolConstant(builder, location);
      return llzkBool_OrBoolOpBuild(builder, location, wrap(cppValue), wrap(cppValue));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `AndBoolOp_build_pass` test
std::unique_ptr<AndBoolOpBuildFuncHelper> AndBoolOpBuildFuncHelper::get() {
  struct Impl : public AndBoolOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      mlir::Value cppValue = CAPITest::cppGenBoolConstant(builder, location);
      return llzkBool_AndBoolOpBuild(builder, location, wrap(cppValue), wrap(cppValue));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `NotBoolOp_build_pass` test
std::unique_ptr<NotBoolOpBuildFuncHelper> NotBoolOpBuildFuncHelper::get() {
  struct Impl : public NotBoolOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      mlir::Value cppValue = CAPITest::cppGenBoolConstant(builder, location);
      return llzkBool_NotBoolOpBuild(builder, location, wrap(cppValue));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `XorBoolOp_build_pass` test
std::unique_ptr<XorBoolOpBuildFuncHelper> XorBoolOpBuildFuncHelper::get() {
  struct Impl : public XorBoolOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      testClass.setAllowNonNativeFieldOpsAttrOnFuncDef(builder);
      mlir::Value cppValue = CAPITest::cppGenBoolConstant(builder, location);
      return llzkBool_XorBoolOpBuild(builder, location, wrap(cppValue), wrap(cppValue));
    }
  };
  return std::make_unique<Impl>();
}
