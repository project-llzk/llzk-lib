//===-- Poly.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Poly.h"

#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Polymorphic/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Polymorphic/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Polymorphic/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_type_var_type_get) {
  auto t = llzkPoly_TypeVarTypeGetFromStringRef(context, mlirStringRefCreateFromCString("T"));
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_type_var_type_pass) {
  auto t = llzkPoly_TypeVarTypeGetFromStringRef(context, mlirStringRefCreateFromCString("T"));
  EXPECT_TRUE(llzkTypeIsA_Poly_TypeVarType(t));
}

TEST_F(CAPITest, llzk_type_var_type_get_from_attr) {
  auto s = mlirStringAttrGet(context, mlirStringRefCreateFromCString("T"));
  auto t = llzkPoly_TypeVarTypeGetFromAttr(s);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_var_type_get_name_ref) {
  auto s = mlirStringRefCreateFromCString("T");
  auto t = llzkPoly_TypeVarTypeGetFromStringRef(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirStringRefEqual(s, llzkPoly_TypeVarTypeGetRefName(t)));
}

TEST_F(CAPITest, llzk_type_var_type_get_name) {
  auto s = mlirStringRefCreateFromCString("T");
  auto t = llzkPoly_TypeVarTypeGetFromStringRef(context, s);
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(sym, llzkPoly_TypeVarTypeGetNameRef(t)));
}

struct ApplyMapOpBuildFuncHelper : public TestAnyBuildFuncHelper<CAPITest> {
  bool callIsA(MlirOperation op) override { return llzkOperationIsA_Poly_ApplyMapOp(op); }
};

TEST_F(CAPITest, llzk_apply_map_op_build) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      auto affine_map_attr = mlirAffineMapAttrGet(affine_map);
      return llzkPoly_ApplyMapOpBuild(
          builder, location, affine_map_attr,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_map) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPoly_ApplyMapOpBuildWithAffineMap(
          builder, location, affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_expr) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto expr = mlirAffineConstantExprGet(testClass.context, 1);
      return llzkPoly_ApplyMapOpBuildWithAffineExpr(
          builder, location, expr,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_op_is_a_apply_map_op_pass) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      auto expr = mlirAffineConstantExprGet(testClass.context, 1);
      return llzkPoly_ApplyMapOpBuildWithAffineExpr(
          builder, location, expr,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_get_affine_map) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirAffineMap affine_map;

    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      this->affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPoly_ApplyMapOpBuildWithAffineMap(
          builder, location, this->affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
    void doOtherChecks(MlirOperation op) override {
      auto out_affine_map = llzkPoly_ApplyMapOpGetAffineMap(op);
      EXPECT_TRUE(mlirAffineMapEqual(this->affine_map, out_affine_map));
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_get_dim_operands) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPoly_ApplyMapOpBuildWithAffineMap(
          builder, location, affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
    void doOtherChecks(MlirOperation op) override {
      auto n_dims = llzkPoly_ApplyMapOpGetNumDimOperands(op);
      llvm::SmallVector<MlirValue> dims(n_dims, MlirValue {.ptr = (void *)NULL});
      llzkPoly_ApplyMapOpGetDimOperands(op, dims.data());
      EXPECT_EQ(dims.size(), 0);
    }
  } helper;
  helper.run(*this);
}

TEST_F(CAPITest, llzk_apply_map_op_get_symbol_operands) {
  struct : ApplyMapOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      llvm::SmallVector<MlirAffineExpr> exprs = {mlirAffineConstantExprGet(testClass.context, 1)};
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkPoly_ApplyMapOpBuildWithAffineMap(
          builder, location, affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
    void doOtherChecks(MlirOperation op) override {
      auto n_syms = llzkPoly_ApplyMapOpGetNumSymbolOperands(op);
      llvm::SmallVector<MlirValue> syms(n_syms, {.ptr = (void *)NULL});
      llzkPoly_ApplyMapOpGetSymbolOperands(op, syms.data());
      EXPECT_EQ(syms.size(), 0);
    }
  } helper;
  helper.run(*this);
}

// Implementation for `ConstReadOp_build_pass` test
std::unique_ptr<ConstReadOpBuildFuncHelper> ConstReadOpBuildFuncHelper::get() {
  struct Impl : public ConstReadOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirAttribute attr =
          mlirFlatSymbolRefAttrGet(testClass.context, mlirStringRefCreateFromCString("const_name"));
      return llzkPoly_ConstReadOpBuild(builder, location, testClass.createIndexType(), attr);
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `UnifiableCastOp_build_pass` test
std::unique_ptr<UnifiableCastOpBuildFuncHelper> UnifiableCastOpBuildFuncHelper::get() {
  struct Impl : public UnifiableCastOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::Operation *> forceCleanup;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirOperation op = testClass.createIndexOperation();
      this->forceCleanup = unwrap(op);
      return llzkPoly_UnifiableCastOpBuild(
          builder, location, testClass.createIndexType(), mlirOperationGetResult(op, 0)
      );
    }
  };
  return std::make_unique<Impl>();
}
