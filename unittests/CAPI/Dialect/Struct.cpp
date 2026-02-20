//===-- Struct.cpp ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Struct.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>

#include <llvm/ADT/SmallVector.h>

#include <gtest/gtest.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Struct/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Struct/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Struct/IR/Types.capi.test.cpp.inc"

TEST_F(CAPITest, llzk_struct_type_get) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  auto t = llzkStruct_StructTypeGet(sym);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_struct_type_get_with_array_attr) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  llvm::SmallVector<MlirAttribute> attrs(
      {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
  );
  auto a = mlirArrayAttrGet(context, attrs.size(), attrs.data());
  auto t = llzkStruct_StructTypeGetWithArrayAttr(sym, a);
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_struct_type_get_with_attrs) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  llvm::SmallVector<MlirAttribute> attrs(
      {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
  );
  auto t = llzkStruct_StructTypeGetWithAttrs(sym, attrs.size(), attrs.data());
  EXPECT_NE(t.ptr, (void *)NULL);
}

TEST_F(CAPITest, llzk_type_is_a_struct_type_pass) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  auto t = llzkStruct_StructTypeGet(sym);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(llzkTypeIsA_Struct_StructType(t));
}

TEST_F(CAPITest, llzk_struct_type_get_name) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  auto t = llzkStruct_StructTypeGet(sym);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(sym, llzkStruct_StructTypeGetNameRef(t)));
}

TEST_F(CAPITest, llzk_struct_type_get_params) {
  auto s = mlirStringRefCreateFromCString("T");
  auto sym = mlirFlatSymbolRefAttrGet(context, s);
  llvm::SmallVector<MlirAttribute> attrs(
      {mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("A"))}
  );
  auto a = mlirArrayAttrGet(context, attrs.size(), attrs.data());
  auto t = llzkStruct_StructTypeGetWithArrayAttr(sym, a);
  EXPECT_NE(t.ptr, (void *)NULL);
  EXPECT_TRUE(mlirAttributeEqual(a, llzkStruct_StructTypeGetParams(t)));
}

struct TestOp {
  MlirOperation op;

  ~TestOp() { mlirOperationDestroy(op); }
};

struct StructDefTest : public CAPITest {
  MlirOperation make_struct_def_op() const {
    auto name = mlirStringRefCreateFromCString("struct.def");
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirNamedAttribute> attrs({mlirNamedAttributeGet(
        mlirIdentifierGet(context, mlirStringRefCreateFromCString("sym_name")),
        mlirStringAttrGet(context, mlirStringRefCreateFromCString("S"))
    )});
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddAttributes(&op_state, attrs.size(), attrs.data());
    return mlirOperationCreate(&op_state);
  }

  MlirOperation make_struct_new_op() const {
    auto struct_name = mlirFlatSymbolRefAttrGet(context, mlirStringRefCreateFromCString("S"));
    auto name = mlirStringRefCreateFromCString("struct.new");
    auto location = mlirLocationUnknownGet(context);
    auto result = llzkStruct_StructTypeGet(struct_name);
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddResults(&op_state, 1, &result);
    return mlirOperationCreate(&op_state);
  }

  MlirOperation make_member_def_op() const {
    auto name = mlirStringRefCreateFromCString("struct.member");
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirNamedAttribute> attrs(
        {mlirNamedAttributeGet(
             mlirIdentifierGet(context, mlirStringRefCreateFromCString("sym_name")),
             mlirStringAttrGet(context, mlirStringRefCreateFromCString("S"))
         ),
         mlirNamedAttributeGet(
             mlirIdentifierGet(context, mlirStringRefCreateFromCString("type")),
             mlirTypeAttrGet(createIndexType())
         )}
    );
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddAttributes(&op_state, attrs.size(), attrs.data());
    return mlirOperationCreate(&op_state);
  }

  TestOp test_op() const {
    auto elt_type = createIndexType();
    auto name = mlirStringRefCreateFromCString("arith.constant");
    auto attr_name = mlirIdentifierGet(context, mlirStringRefCreateFromCString("value"));
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirType> results({elt_type});
    auto attr = mlirIntegerAttrGet(elt_type, 1);
    llvm::SmallVector<MlirNamedAttribute> attrs({mlirNamedAttributeGet(attr_name, attr)});
    auto op_state = mlirOperationStateGet(name, location);
    mlirOperationStateAddResults(&op_state, results.size(), results.data());
    mlirOperationStateAddAttributes(&op_state, attrs.size(), attrs.data());
    return {
        .op = mlirOperationCreate(&op_state),
    };
  }
};

TEST_F(StructDefTest, llzk_operation_is_a_struct_def_op_pass) {
  auto op = make_struct_def_op();
  EXPECT_TRUE(llzkOperationIsA_Struct_StructDefOp(op));
}

TEST_F(StructDefTest, llzk_struct_def_op_get_body) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetBody(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_body_region) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetBodyRegion(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_type) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetType(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_type_with_params) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    auto attrs = mlirArrayAttrGet(mlirOperationGetContext(op.op), 0, (const MlirAttribute *)NULL);
    llzkStruct_StructDefOpGetTypeWithParams(op.op, attrs);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_member_def) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    MlirIdentifier name =
        mlirIdentifierGet(mlirOperationGetContext(op.op), mlirStringRefCreateFromCString("p"));
    llzkStruct_StructDefOpGetMemberDef(op.op, name);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_member_defs) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetMemberDefs(op.op, (MlirOperation *)NULL);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_num_member_defs) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetNumMemberDefs(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_has_columns) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpHasColumns(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_compute_func_op) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetComputeFuncOp(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_constrain_func_op) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetConstrainFuncOp(op.op);
  }
}

static char *cmalloc(size_t s) { return (char *)malloc(s); }

TEST_F(StructDefTest, llzk_struct_def_op_get_header_string) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    intptr_t size = 0;
    auto str = llzkStruct_StructDefOpGetHeaderString(op.op, &size, cmalloc);
    free(static_cast<void *>(const_cast<char *>(str)));
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_has_param_name) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    auto name = mlirStringRefCreateFromCString("p");
    llzkStruct_StructDefOpHasParamName(op.op, name);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_get_fully_qualified_name) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpGetFullyQualifiedName(op.op);
  }
}

TEST_F(StructDefTest, llzk_struct_def_op_is_main_component) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_StructDefOp(op.op)) {
    llzkStruct_StructDefOpIsMainComponent(op.op);
  }
}

TEST_F(StructDefTest, llzk_operation_is_a_member_def_op_pass) {
  auto op = make_member_def_op();
  EXPECT_TRUE(llzkOperationIsA_Struct_MemberDefOp(op));
}

TEST_F(StructDefTest, llzk_member_def_op_has_public_attr) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_MemberDefOp(op.op)) {
    llzkStruct_MemberDefOpHasPublicAttr(op.op);
  }
}

TEST_F(StructDefTest, llzk_member_def_op_set_public_attr) {
  auto op = test_op();
  if (llzkOperationIsA_Struct_MemberDefOp(op.op)) {
    llzkStruct_MemberDefOpSetPublicAttr(op.op, true);
  }
}

struct MemberReadOpBuildFuncHelper : public TestAnyBuildFuncHelper<StructDefTest> {
  MlirOperation struct_new_op;
  bool callIsA(MlirOperation op) override { return llzkOperationIsA_Struct_MemberReadOp(op); }
  ~MemberReadOpBuildFuncHelper() override { mlirOperationDestroy(this->struct_new_op); }
};

TEST_F(StructDefTest, llzk_member_read_op_build) {
  struct : MemberReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      return llzkStruct_MemberReadOpBuild(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f")
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(StructDefTest, llzk_member_read_op_build_with_affine_map_distance) {
  struct : MemberReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      llvm::SmallVector<MlirAffineExpr> exprs({mlirAffineConstantExprGet(testClass.context, 1)});
      auto affine_map = mlirAffineMapGet(testClass.context, 0, 0, exprs.size(), exprs.data());
      return llzkStruct_MemberReadOpBuildWithAffineMapDistance(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"),
          affine_map,
          MlirValueRange {
              .values = (const MlirValue *)NULL,
              .size = 0,
          }
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(StructDefTest, llzk_member_read_op_builder_with_const_param_distance) {
  struct : MemberReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      return llzkStruct_MemberReadOpBuildWithConstParamDistance(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"),
          mlirStringRefCreateFromCString("N")
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(StructDefTest, llzk_member_read_op_build_with_literal_distance) {
  struct : MemberReadOpBuildFuncHelper {
    MlirOperation callBuild(
        const StructDefTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      this->struct_new_op = testClass.make_struct_new_op();
      auto index_type = testClass.createIndexType();
      auto struct_value = mlirOperationGetResult(struct_new_op, 0);
      return llzkStruct_MemberReadOpBuildWithLiteralDistance(
          builder, location, index_type, struct_value, mlirStringRefCreateFromCString("f"), 1
      );
    }
  } helper;
  helper.run(*this);
}

// Implementation for `CreateStructOp_build_pass` test
std::unique_ptr<CreateStructOpBuildFuncHelper> CreateStructOpBuildFuncHelper::get() {
  struct Impl : public CreateStructOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Type structType;
      {
        // Use "@compute" function as parent to avoid the following:
        // error: 'struct.new' op only valid within a 'function.def' with 'function.allow_witness'
        this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
            builder, location, llzk::function::FunctionKind::StructCompute
        );
        structType = llzk::component::StructType::get(
            mlir::FlatSymbolRefAttr::get(unwrap(testClass.context), "StructName")
        );
      }
      return llzkStruct_CreateStructOpBuild(builder, location, wrap(structType));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `StructDefOp_build_pass` test
std::unique_ptr<StructDefOpBuildFuncHelper> StructDefOpBuildFuncHelper::get() {
  struct Impl : public StructDefOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use ModuleOp as parent to avoid the following:
      // error: 'struct.def' op expects parent op 'builtin.module'
      auto name = "TestStruct";
      this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
      auto result = llzkStruct_StructDefOpBuild(
          builder, location,
          mlirIdentifierGet(testClass.context, mlirStringRefCreateFromCString(name)),
          MlirAttribute {}
      );
      // Populate the struct to avoid the errors mentioned below.
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        mlir::Location cppLoc = unwrap(location);
        auto structDefOp = mlir::unwrap_cast<llzk::component::StructDefOp>(result);
        // error: 'struct.def' op region #0 ('bodyRegion') failed to verify constraint:
        //        region with 1 blocks
        (void)structDefOp.getBodyRegion().emplaceBlock();
        // error: 'struct.def' op must define either only a "@product" function, or both "@compute"
        //        and "@constrain" functions; could not find "@product", "@compute", or "@constrain"
        auto fn = llzk::ModuleBuilder::buildProductFn(structDefOp, cppLoc);
        // error: empty block: expect at least a terminator
        mlir::OpBuilder bldr(fn.getBody());
        auto v = bldr.create<llzk::component::CreateStructOp>(cppLoc, structDefOp.getType());
        bldr.create<llzk::function::ReturnOp>(cppLoc, mlir::ValueRange {v});
      }
      return result;
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `MemberWriteOp_build_pass` test
std::unique_ptr<MemberWriteOpBuildFuncHelper> MemberWriteOpBuildFuncHelper::get() {
  struct Impl : public MemberWriteOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Value component;
      mlir::Attribute memberNameAttr;
      {
        // Use "@compute" function as parent to avoid the following:
        // error: 'struct.new' op only valid within a 'function.def' with 'function.allow_witness'
        this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
            builder, location, llzk::function::FunctionKind::StructCompute
        );
        mlir::Location cppLoc = unwrap(location);
        mlir::OpBuilder *bldr = unwrap(builder);
        component = bldr->create<llzk::component::CreateStructOp>(
            cppLoc, bldr->getInsertionBlock()
                        ->getParentOp()
                        ->getParentOfType<llzk::component::StructDefOp>()
                        .getType()
        );
        memberNameAttr = mlir::FlatSymbolRefAttr::get(unwrap(testClass.context), "MemberName");
      }
      return llzkStruct_MemberWriteOpBuild(
          builder, location, wrap(component), wrap(component), wrap(memberNameAttr)
      );
    }
  };
  return std::make_unique<Impl>();
}
