//===-- Poly.cpp ------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/Compare.h"

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
      auto affine_map = mlirAffineMapGet(
          testClass.context, 0, 0, llzk::checkedCast<intptr_t>(exprs.size()), exprs.data()
      );
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
      auto affine_map = mlirAffineMapGet(
          testClass.context, 0, 0, llzk::checkedCast<intptr_t>(exprs.size()), exprs.data()
      );
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
      this->affine_map = mlirAffineMapGet(
          testClass.context, 0, 0, llzk::checkedCast<intptr_t>(exprs.size()), exprs.data()
      );
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
      auto affine_map = mlirAffineMapGet(
          testClass.context, 0, 0, llzk::checkedCast<intptr_t>(exprs.size()), exprs.data()
      );
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
      auto affine_map = mlirAffineMapGet(
          testClass.context, 0, 0, llzk::checkedCast<intptr_t>(exprs.size()), exprs.data()
      );
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

// Implementation for `TemplateOp_build_pass` test
std::unique_ptr<TemplateOpBuildFuncHelper> TemplateOpBuildFuncHelper::get() {
  struct Impl : public TemplateOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
      auto result = llzkPoly_TemplateOpBuild(
          builder, location,
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString("template_name"))
      );
      // Additional initialization to avoid the errors mentioned below.
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        auto templateOp = mlir::unwrap_cast<llzk::polymorphic::TemplateOp>(result);
        // error: 'poly.template' op region #0 ('bodyRegion') failed to verify constraint: region
        // with 1 blocks
        (void)templateOp.getBodyRegion().emplaceBlock();
      }
      return result;
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `TemplateParamOp_build_pass` test
std::unique_ptr<TemplateParamOpBuildFuncHelper> TemplateParamOpBuildFuncHelper::get() {
  struct Impl : public TemplateParamOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {

      // Needs parent `poly.template` for verifier to pass.
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
        mlir::OpBuilder *cppBuilder = unwrap(builder);
        auto polyTemplate = cppBuilder->create<llzk::polymorphic::TemplateOp>(
            unwrap(location), mlir::StringAttr::get(unwrap(testClass.context), "template_name")
        );
        cppBuilder->setInsertionPointToStart(&polyTemplate.getBodyRegion().emplaceBlock());
      }
      return llzkPoly_TemplateParamOpBuild(
          builder, location,
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString("param_name")),
          MlirAttribute()
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `TemplateExprOp_build_pass` test
std::unique_ptr<TemplateExprOpBuildFuncHelper> TemplateExprOpBuildFuncHelper::get() {
  struct Impl : public TemplateExprOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {

      // Needs parent `poly.template` for verifier to pass.
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
        mlir::OpBuilder *cppBuilder = unwrap(builder);
        auto polyTemplate = cppBuilder->create<llzk::polymorphic::TemplateOp>(
            unwrap(location), mlir::StringAttr::get(unwrap(testClass.context), "template_name")
        );
        cppBuilder->setInsertionPointToStart(&polyTemplate.getBodyRegion().emplaceBlock());
      }
      auto retVal = llzkPoly_TemplateExprOpBuild(
          builder, location,
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString("expr_name"))
      );
      // Needs region with 1 block and a `yield` op for verifier to pass.
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        auto exprOp = mlir::unwrap_cast<llzk::polymorphic::TemplateExprOp>(retVal);
        mlir::OpBuilder *cppBuilder = unwrap(builder);
        cppBuilder->setInsertionPointToStart(&exprOp.getInitializerRegion().emplaceBlock());
        mlir::Value val = testClass.cppGenFeltConstant(builder, location);
        cppBuilder->create<llzk::polymorphic::YieldOp>(unwrap(location), val);
      }
      return retVal;
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `YieldOp_build_pass` test
std::unique_ptr<YieldOpBuildFuncHelper> YieldOpBuildFuncHelper::get() {
  struct Impl : public YieldOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {

      // Needs parent `poly.param` in `poly.template` for verifier to pass.
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
        mlir::OpBuilder *cppBuilder = unwrap(builder);
        auto polyTemplate = cppBuilder->create<llzk::polymorphic::TemplateOp>(
            unwrap(location), mlir::StringAttr::get(unwrap(testClass.context), "template_name")
        );
        cppBuilder->setInsertionPointToStart(&polyTemplate.getBodyRegion().emplaceBlock());
        auto templateExpr = cppBuilder->create<llzk::polymorphic::TemplateExprOp>(
            unwrap(location), mlir::StringAttr::get(unwrap(testClass.context), "expr_name")
        );
        cppBuilder->setInsertionPointToStart(&templateExpr.getInitializerRegion().emplaceBlock());
      }
      mlir::Value val = testClass.cppGenFeltConstant(builder, location);
      return llzkPoly_YieldOpBuild(builder, location, wrap(val));
    }
  };
  return std::make_unique<Impl>();
}

//===----------------------------------------------------------------------===//
// TemplateOp hasConstOps / numConstOps / getConstNames / hasConstNamed tests
//===----------------------------------------------------------------------===//

/// Helper: creates a TemplateOp in its own module and emplaces its body block.
/// Returns the TemplateOp; the module owning ref is written to `outModule` to
/// keep it alive for the duration of the test.
static llzk::polymorphic::TemplateOp createTemplateWithBlock(
    const CAPITest &tc, MlirOpBuilder builder, MlirLocation location,
    mlir::OwningOpRef<mlir::ModuleOp> &outModule
) {
  outModule = tc.cppNewModuleAndSetInsertionPoint(builder, location);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  auto tmpl = cppBuilder->create<llzk::polymorphic::TemplateOp>(
      unwrap(location), mlir::StringAttr::get(unwrap(tc.context), "T")
  );
  cppBuilder->setInsertionPointToStart(&tmpl.getBodyRegion().emplaceBlock());
  return tmpl;
}

// --- TemplateParamOp tests ---

TEST_F(CAPITest, llzk_template_op_has_param_ops_empty) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  EXPECT_FALSE(llzkPoly_TemplateOpHasConstParamOps(wrap(tmpl.getOperation())));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_has_param_ops_nonempty) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  cppBuilder->create<llzk::polymorphic::TemplateParamOp>(
      unwrap(loc), mlir::StringAttr::get(unwrap(context), "P"), mlir::TypeAttr()
  );
  EXPECT_TRUE(llzkPoly_TemplateOpHasConstParamOps(wrap(tmpl.getOperation())));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_num_param_ops_zero) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  EXPECT_EQ(llzkPoly_TemplateOpNumConstParamOps(wrap(tmpl.getOperation())), 0);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_num_param_ops_two) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  for (const char *name : {"N", "M"}) {
    cppBuilder->create<llzk::polymorphic::TemplateParamOp>(
        unwrap(loc), mlir::StringAttr::get(unwrap(context), name), mlir::TypeAttr()
    );
  }
  EXPECT_EQ(llzkPoly_TemplateOpNumConstParamOps(wrap(tmpl.getOperation())), 2);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_get_param_names) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  for (const char *name : {"N", "M"}) {
    cppBuilder->create<llzk::polymorphic::TemplateParamOp>(
        unwrap(loc), mlir::StringAttr::get(unwrap(context), name), mlir::TypeAttr()
    );
  }
  MlirOperation op = wrap(tmpl.getOperation());
  auto n = llzkPoly_TemplateOpNumConstParamOps(op);
  ASSERT_EQ(n, 2);
  llvm::SmallVector<MlirAttribute> names(n);
  llzkPoly_TemplateOpGetConstParamNames(op, names.data());
  EXPECT_TRUE(mlirAttributeEqual(
      names[0], mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("N"))
  ));
  EXPECT_TRUE(mlirAttributeEqual(
      names[1], mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("M"))
  ));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_has_param_named_found) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  cppBuilder->create<llzk::polymorphic::TemplateParamOp>(
      unwrap(loc), mlir::StringAttr::get(unwrap(context), "P"), mlir::TypeAttr()
  );
  MlirOperation op = wrap(tmpl.getOperation());
  EXPECT_TRUE(llzkPoly_TemplateOpHasConstParamNamed(op, mlirStringRefCreateFromCString("P")));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_has_param_named_not_found) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  cppBuilder->create<llzk::polymorphic::TemplateParamOp>(
      unwrap(loc), mlir::StringAttr::get(unwrap(context), "P"), mlir::TypeAttr()
  );
  MlirOperation op = wrap(tmpl.getOperation());
  EXPECT_FALSE(llzkPoly_TemplateOpHasConstParamNamed(op, mlirStringRefCreateFromCString("Q")));
  mlirOpBuilderDestroy(builder);
}

// --- TemplateExprOp tests ---

/// Helper: adds a TemplateExprOp with a felt constant yield to the current insertion point,
/// then restores the insertion point back to the template body for subsequent ops.
static llzk::polymorphic::TemplateExprOp addTemplateExprOp(
    const CAPITest &tc, MlirOpBuilder builder, MlirLocation location, const char *name
) {
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  auto exprOp = cppBuilder->create<llzk::polymorphic::TemplateExprOp>(
      unwrap(location), mlir::StringAttr::get(unwrap(tc.context), name)
  );
  cppBuilder->setInsertionPointToStart(&exprOp.getInitializerRegion().emplaceBlock());
  mlir::Value val = tc.cppGenFeltConstant(builder, location);
  cppBuilder->create<llzk::polymorphic::YieldOp>(unwrap(location), val);
  cppBuilder->setInsertionPointToEnd(&exprOp->getParentRegion()->back());
  return exprOp;
}

TEST_F(CAPITest, llzk_template_op_has_expr_ops_empty) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  EXPECT_FALSE(llzkPoly_TemplateOpHasConstExprOps(wrap(tmpl.getOperation())));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_has_expr_ops_nonempty) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  addTemplateExprOp(*this, builder, loc, "E");
  EXPECT_TRUE(llzkPoly_TemplateOpHasConstExprOps(wrap(tmpl.getOperation())));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_num_expr_ops_zero) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  EXPECT_EQ(llzkPoly_TemplateOpNumConstExprOps(wrap(tmpl.getOperation())), 0);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_num_expr_ops_two) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  addTemplateExprOp(*this, builder, loc, "E1");
  addTemplateExprOp(*this, builder, loc, "E2");
  EXPECT_EQ(llzkPoly_TemplateOpNumConstExprOps(wrap(tmpl.getOperation())), 2);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_get_expr_names) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  addTemplateExprOp(*this, builder, loc, "E1");
  addTemplateExprOp(*this, builder, loc, "E2");
  MlirOperation op = wrap(tmpl.getOperation());
  auto n = llzkPoly_TemplateOpNumConstExprOps(op);
  ASSERT_EQ(n, 2);
  llvm::SmallVector<MlirAttribute> names(n);
  llzkPoly_TemplateOpGetConstExprNames(op, names.data());
  EXPECT_TRUE(mlirAttributeEqual(
      names[0], mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("E1"))
  ));
  EXPECT_TRUE(mlirAttributeEqual(
      names[1], mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("E2"))
  ));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_has_expr_named_found) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  addTemplateExprOp(*this, builder, loc, "E");
  MlirOperation op = wrap(tmpl.getOperation());
  EXPECT_TRUE(llzkPoly_TemplateOpHasConstExprNamed(op, mlirStringRefCreateFromCString("E")));
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_has_expr_named_not_found) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  addTemplateExprOp(*this, builder, loc, "E");
  MlirOperation op = wrap(tmpl.getOperation());
  EXPECT_FALSE(llzkPoly_TemplateOpHasConstExprNamed(op, mlirStringRefCreateFromCString("F")));
  mlirOpBuilderDestroy(builder);
}

// --- Cross-type isolation: param ops don't count as expr ops and vice versa ---

TEST_F(CAPITest, llzk_template_op_param_does_not_affect_expr_count) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  mlir::OpBuilder *cppBuilder = unwrap(builder);
  cppBuilder->create<llzk::polymorphic::TemplateParamOp>(
      unwrap(loc), mlir::StringAttr::get(unwrap(context), "P"), mlir::TypeAttr()
  );
  MlirOperation op = wrap(tmpl.getOperation());
  EXPECT_TRUE(llzkPoly_TemplateOpHasConstParamOps(op));
  EXPECT_FALSE(llzkPoly_TemplateOpHasConstExprOps(op));
  EXPECT_EQ(llzkPoly_TemplateOpNumConstExprOps(op), 0);
  mlirOpBuilderDestroy(builder);
}

TEST_F(CAPITest, llzk_template_op_expr_does_not_affect_param_count) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation loc = mlirLocationUnknownGet(context);
  mlir::OwningOpRef<mlir::ModuleOp> module;
  auto tmpl = createTemplateWithBlock(*this, builder, loc, module);
  addTemplateExprOp(*this, builder, loc, "E");
  MlirOperation op = wrap(tmpl.getOperation());
  EXPECT_FALSE(llzkPoly_TemplateOpHasConstParamOps(op));
  EXPECT_EQ(llzkPoly_TemplateOpNumConstParamOps(op), 0);
  EXPECT_TRUE(llzkPoly_TemplateOpHasConstExprOps(op));
  mlirOpBuilderDestroy(builder);
}
