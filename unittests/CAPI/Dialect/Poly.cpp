//===-- Poly.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Poly.h"

#include <mlir-c/BuiltinAttributes.h>

#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"

TEST_F(CAPITest, mlir_get_dialect_handle_llzk_polymorphic) {
  (void)mlirGetDialectHandle__llzk__polymorphic__();
}

TEST_F(CAPITest, llzk_type_var_type_get) {
  auto t = llzkTypeVarTypeGet(context, mlirStringRefCreateFromCString("T"));
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_type_var_type_pass) {
  auto t = llzkTypeVarTypeGet(context, mlirStringRefCreateFromCString("T"));
  EXPECT_TRUE(llzkTypeIsA_Poly_TypeVarType(t));
}

TEST_F(CAPITest, llzk_type_var_type_get_from_attr) {
  auto s = mlirStringAttrGet(context, mlirStringRefCreateFromCString("T"));
  auto t = llzkTypeVarTypeGetFromAttr(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_var_type_get_name_ref) {
  auto s = mlirStringRefCreateFromCString("T");
  auto t = llzkTypeVarTypeGet(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirStringRefEqual(s, llzkTypeVarTypeGetNameRef(t)));
}

TEST_F(CAPITest, llzk_type_var_type_get_name) {
  auto s = mlirStringRefCreateFromCString("T");
  auto t = llzkTypeVarTypeGet(context, s);
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(sym, llzkTypeVarTypeGetName(t)));
}

TEST_F(CAPITest, llzk_apply_map_op_build) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(context, 1)});
  auto affine_map = mlirAffineMapGet(context, 0, 0, exprs.size(), exprs.data());
  auto affine_map_attr = mlirAffineMapAttrGet(affine_map);
  auto op = llzkPoly_ApplyMapOpBuild(
      builder, location, affine_map_attr,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_map) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(context, 1)});
  auto affine_map = mlirAffineMapGet(context, 0, 0, exprs.size(), exprs.data());
  auto op = llzkPoly_ApplyMapOpBuildWithAffineMap(
      builder, location, affine_map,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_apply_map_op_build_with_affine_expr) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  auto expr = mlirAffineConstantExprGet(context, 1);
  auto op = llzkPoly_ApplyMapOpBuildWithAffineExpr(
      builder, location, expr,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_op_is_a_apply_map_op_pass) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  auto expr = mlirAffineConstantExprGet(context, 1);
  auto op = llzkPoly_ApplyMapOpBuildWithAffineExpr(
      builder, location, expr,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  EXPECT_TRUE(llzkOperationIsA_Poly_ApplyMapOp(op));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_apply_map_op_get_affine_map) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(context, 1)});
  auto affine_map = mlirAffineMapGet(context, 0, 0, exprs.size(), exprs.data());
  auto op = llzkPoly_ApplyMapOpBuildWithAffineMap(
      builder, location, affine_map,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  auto out_affine_map = llzkApplyMapOpGetAffineMap(op);
  EXPECT_TRUE(mlirAffineMapEqual(affine_map, out_affine_map));
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_apply_map_op_get_dim_operands) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(context, 1)});
  auto affine_map = mlirAffineMapGet(context, 0, 0, exprs.size(), exprs.data());
  auto op = llzkPoly_ApplyMapOpBuildWithAffineMap(
      builder, location, affine_map,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  auto n_dims = llzkApplyMapOpGetNumDimOperands(op);
  llvm::SmallVector<MlirValue> dims(n_dims, MlirValue {.ptr = (void *)NULL});
  llzkApplyMapOpGetDimOperands(op, dims.data());
  EXPECT_EQ(dims.size(), 0);
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_apply_map_op_get_symbol_operands) {
  auto builder = mlirOpBuilderCreate(context);
  auto location = mlirLocationUnknownGet(context);
  llvm::SmallVector<MlirAffineExpr> exprs = {mlirAffineConstantExprGet(context, 1)};
  auto affine_map = mlirAffineMapGet(context, 0, 0, exprs.size(), exprs.data());
  auto op = llzkPoly_ApplyMapOpBuildWithAffineMap(
      builder, location, affine_map,
      MlirValueRange {
          .values = (const MlirValue *)NULL,
          .size = 0,
      }
  );
  EXPECT_NE(op.ptr, (void *)NULL);
  EXPECT_TRUE(mlirOperationVerify(op));
  auto n_syms = llzkApplyMapOpGetNumSymbolOperands(op);
  llvm::SmallVector<MlirValue> syms(n_syms, {.ptr = (void *)NULL});
  llzkApplyMapOpGetSymbolOperands(op, syms.data());
  EXPECT_EQ(syms.size(), 0);
  mlirOperationDestroy(op);
  mlirOpBuilderDestroy(builder);
}
