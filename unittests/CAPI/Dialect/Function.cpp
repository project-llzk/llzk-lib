//===-- Function.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Function.h"

#include "llzk-c/Support.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include "../CAPITestBase.h"

static MlirType
create_func_type(MlirContext ctx, llvm::ArrayRef<MlirType> ins, llvm::ArrayRef<MlirType> outs) {
  return mlirFunctionTypeGet(ctx, ins.size(), ins.data(), outs.size(), outs.data());
}

static MlirOperation create_func_def_op(
    MlirContext ctx, const char *name, MlirType type, llvm::ArrayRef<MlirNamedAttribute> attrs,
    llvm::ArrayRef<MlirAttribute> arg_attrs
) {
  auto location = mlirLocationUnknownGet(ctx);
  return llzkFunction_FuncDefOpCreateWithAttrsAndArgAttrs(
      location, mlirStringRefCreateFromCString(name), type, attrs.size(), attrs.data(),
      arg_attrs.size(), arg_attrs.data()
  );
}

template <int64_t N> static llvm::SmallVector<MlirAttribute, N> empty_arg_attrs(MlirContext ctx) {
  return llvm::SmallVector<MlirAttribute, N>(
      N, mlirDictionaryAttrGet(ctx, 0, (const MlirNamedAttribute *)NULL)
  );
}

struct TestFuncDefOp {
  llvm::SmallVector<MlirType> in_types, out_types;
  llvm::StringRef name;
  MlirOperation op;

  MlirStringRef nameRef() const { return {.data = name.data(), .length = name.size()}; }

  ~TestFuncDefOp() { mlirOperationDestroy(op); }
};

class FuncDialectTest : public CAPITest {

protected:
  TestFuncDefOp test_function() {
    auto in_types =
        llvm::SmallVector<MlirType>({mlirIndexTypeGet(context), mlirIndexTypeGet(context)});
    auto in_attrs = empty_arg_attrs<2>(context);
    auto out_types = llvm::SmallVector<MlirType>({mlirIndexTypeGet(context)});
    const auto *name = "foo";
    return {
        .in_types = in_types,
        .out_types = out_types,
        .name = name,
        .op = create_func_def_op(
            context, name, create_func_type(context, in_types, out_types),
            llvm::ArrayRef<MlirNamedAttribute>(), in_attrs
        ),
    };
  }

  TestFuncDefOp test_function0() {
    auto in_types = llvm::SmallVector<MlirType>();
    auto out_types = llvm::SmallVector<MlirType>({mlirIndexTypeGet(context)});
    const auto *name = "bar";
    return {
        .in_types = in_types,
        .out_types = out_types,
        .name = name,
        .op = create_func_def_op(
            context, name, create_func_type(context, in_types, out_types),
            llvm::ArrayRef<MlirNamedAttribute>(), llvm::ArrayRef<MlirAttribute>()
        ),
    };
  }
};

TEST_F(FuncDialectTest, mlir_get_dialect_handle_llzk_function) {
  (void)mlirGetDialectHandle__llzk__function__();
}

TEST_F(FuncDialectTest, llzk_func_def_op_create_with_attrs_and_arg_attrs) {
  MlirType in_types[] = {mlirIndexTypeGet(context)};
  auto in_attrs = empty_arg_attrs<1>(context);
  auto op = create_func_def_op(
      context, "foo",
      create_func_type(context, llvm::ArrayRef(in_types, 1), llvm::ArrayRef<MlirType>()),
      llvm::ArrayRef<MlirNamedAttribute>(), in_attrs
  );
  mlirOperationDestroy(op);
}

TEST_F(FuncDialectTest, llzk_operation_is_a_func_def_op_pass) {
  auto f = test_function();
  EXPECT_TRUE(llzkOperationIsA_Function_FuncDefOp(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_get_has_allow_constraint_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowConstraintAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_set_allow_constraint_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowConstraintAttr(f.op));
  llzkFunction_FuncDefOpSetAllowConstraintAttr(f.op, true);
  EXPECT_TRUE(llzkFunction_FuncDefOpGetHasAllowConstraintAttr(f.op));
  llzkFunction_FuncDefOpSetAllowConstraintAttr(f.op, false);
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowConstraintAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_get_has_allow_witness_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowWitnessAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_set_allow_witness_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowWitnessAttr(f.op));
  llzkFunction_FuncDefOpSetAllowWitnessAttr(f.op, true);
  EXPECT_TRUE(llzkFunction_FuncDefOpGetHasAllowWitnessAttr(f.op));
  llzkFunction_FuncDefOpSetAllowWitnessAttr(f.op, false);
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowWitnessAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_get_has_allow_non_native_field_ops_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowNonNativeFieldOpsAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_set_allow_non_native_field_ops_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowNonNativeFieldOpsAttr(f.op));
  llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(f.op, true);
  EXPECT_TRUE(llzkFunction_FuncDefOpGetHasAllowNonNativeFieldOpsAttr(f.op));
  llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(f.op, false);
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasAllowNonNativeFieldOpsAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_get_has_arg_is_pub) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpGetHasArgIsPub(f.op, 0));
}

TEST_F(FuncDialectTest, llzk_func_def_op_get_fully_qualified_name) {
  // Because the func is not included in a module or struct calling this method will result
  // in an error. To avoid this while still having a test that links against the function we
  // only "call" the method on a condition that is actually impossible but the compiler
  // cannot see that.
  auto f = test_function();
  if (f.op.ptr == (void *)NULL) {
    llzkFunction_FuncDefOpGetFullyQualifiedName(f.op);
  }
}

#define false_pred_test(name, func)                                                                \
  TEST_F(FuncDialectTest, name) {                                                                  \
    auto f = test_function();                                                                      \
    EXPECT_FALSE(func(f.op));                                                                      \
  }

false_pred_test(llzk_func_def_op_get_name_is_compute, llzkFunction_FuncDefOpGetNameIsCompute);
false_pred_test(llzk_func_def_op_get_name_is_constrain, llzkFunction_FuncDefOpGetNameIsConstrain);
false_pred_test(llzk_func_def_op_get_is_in_struct, llzkFunction_FuncDefOpGetIsInStruct);
false_pred_test(llzk_func_def_op_get_is_struct_compute, llzkFunction_FuncDefOpGetIsStructCompute);
false_pred_test(
    llzk_func_def_op_get_is_struct_constrain, llzkFunction_FuncDefOpGetIsStructConstrain
);

/// We want to link the function to make sure it has been implemented but we don't want to
/// call it because the precondition checks will fail with the test function.
#define link_test_func_def_op(name, func)                                                          \
  TEST_F(FuncDialectTest, name) {                                                                  \
    auto f = test_function();                                                                      \
    if (llzkFunction_FuncDefOpGetIsInStruct(f.op)) {                                               \
      func(f.op);                                                                                  \
    }                                                                                              \
  }

link_test_func_def_op(
    llzk_func_def_op_get_self_value_from_compute, llzkFunction_FuncDefOpGetSelfValueFromCompute
);
link_test_func_def_op(
    llzk_func_def_op_get_self_value_from_constrain, llzkFunction_FuncDefOpGetSelfValueFromConstrain
);
link_test_func_def_op(
    llzk_func_def_op_get_single_result_type_of_compute,
    llzkFunction_FuncDefOpGetSingleResultTypeOfCompute
);

TEST_F(FuncDialectTest, llzk_call_op_build) {
  auto f = test_function0();
  auto ctx = mlirOperationGetContext(f.op);
  auto builder = mlirOpBuilderCreate(ctx);
  auto location = mlirLocationUnknownGet(ctx);
  auto callee_name = mlirFlatSymbolRefAttrGet(ctx, f.nameRef());
  auto call = llzkFunction_CallOpBuild(
      builder, location, f.out_types.size(), f.out_types.data(), callee_name, 0,
      (const MlirValue *)NULL
  );
  EXPECT_TRUE(mlirOperationVerify(call));
  mlirOperationDestroy(call);
  mlirOpBuilderDestroy(builder);
}

TEST_F(FuncDialectTest, llzk_call_op_build_to_callee) {
  auto f = test_function0();
  auto ctx = mlirOperationGetContext(f.op);
  auto builder = mlirOpBuilderCreate(ctx);
  auto location = mlirLocationUnknownGet(ctx);
  auto call = llzkFunction_CallOpBuildToCallee(builder, location, f.op, 0, (const MlirValue *)NULL);
  EXPECT_TRUE(mlirOperationVerify(call));
  mlirOperationDestroy(call);
  mlirOpBuilderDestroy(builder);
}

TEST_F(FuncDialectTest, llzk_call_op_build_with_map_operands) {
  auto f = test_function0();
  auto ctx = mlirOperationGetContext(f.op);
  auto builder = mlirOpBuilderCreate(ctx);
  auto location = mlirLocationUnknownGet(ctx);
  auto callee_name = mlirFlatSymbolRefAttrGet(ctx, f.nameRef());
  auto mapOperands = llzkAffineMapOperandsBuilderCreate();
  auto call = llzkFunction_CallOpBuildWithMapOperands(
      builder, location, f.out_types.size(), f.out_types.data(), callee_name, mapOperands, 0,
      (const MlirValue *)NULL
  );
  EXPECT_TRUE(mlirOperationVerify(call));
  mlirOperationDestroy(call);
  llzkAffineMapOperandsBuilderDestroy(&mapOperands);
  mlirOpBuilderDestroy(builder);
}

TEST_F(FuncDialectTest, llzk_call_op_build_to_callee_with_map_operands) {
  auto f = test_function0();
  auto ctx = mlirOperationGetContext(f.op);
  auto builder = mlirOpBuilderCreate(ctx);
  auto location = mlirLocationUnknownGet(ctx);
  auto mapOperands = llzkAffineMapOperandsBuilderCreate();
  auto call = llzkFunction_CallOpBuildToCalleeWithMapOperands(
      builder, location, f.op, mapOperands, 0, (const MlirValue *)NULL
  );
  EXPECT_TRUE(mlirOperationVerify(call));
  mlirOperationDestroy(call);
  llzkAffineMapOperandsBuilderDestroy(&mapOperands);
  mlirOpBuilderDestroy(builder);
}

#define call_pred_test(name, func, expected)                                                       \
  TEST_F(FuncDialectTest, name) {                                                                  \
    auto f = test_function0();                                                                     \
    auto builder = mlirOpBuilderCreate(context);                                                   \
    auto location = mlirLocationUnknownGet(context);                                               \
    auto call =                                                                                    \
        llzkFunction_CallOpBuildToCallee(builder, location, f.op, 0, (const MlirValue *)NULL);     \
    EXPECT_EQ(func(call), expected);                                                               \
    mlirOperationDestroy(call);                                                                    \
    mlirOpBuilderDestroy(builder);                                                                 \
  }

call_pred_test(test_llzk_operation_is_a_call_op_pass, llzkOperationIsA_Function_CallOp, true);

TEST_F(FuncDialectTest, llzk_call_op_get_callee_type) {
  auto f = test_function0();
  auto ctx = mlirOperationGetContext(f.op);
  auto builder = mlirOpBuilderCreate(ctx);
  auto location = mlirLocationUnknownGet(ctx);
  auto call = llzkFunction_CallOpBuildToCallee(builder, location, f.op, 0, (const MlirValue *)NULL);

  auto func_type = create_func_type(ctx, f.in_types, f.out_types);
  auto out_type = llzkFunction_CallOpGetCalleeType(call);
  EXPECT_TRUE(mlirTypeEqual(func_type, out_type));

  mlirOperationDestroy(call);
  mlirOpBuilderDestroy(builder);
}

call_pred_test(
    test_llzk_call_op_get_callee_is_compute, llzkFunction_CallOpGetCalleeIsCompute, false
);
call_pred_test(
    test_llzk_call_op_get_callee_is_constrain, llzkFunction_CallOpGetCalleeIsConstrain, false
);
call_pred_test(
    test_llzk_call_op_get_callee_is_struct_compute, llzkFunction_CallOpGetCalleeIsStructCompute,
    false
);
call_pred_test(
    test_llzk_call_op_get_callee_is_struct_constrain, llzkFunction_CallOpGetCalleeIsStructConstrain,
    false
);

/// We want to link the function to make sure it has been implemented but we don't want to
/// call it because the precondition checks will fail with the test function.
#define link_test_call_op(name, func)                                                              \
  TEST_F(FuncDialectTest, name) {                                                                  \
    auto f = test_function0();                                                                     \
    auto builder = mlirOpBuilderCreate(context);                                                   \
    auto location = mlirLocationUnknownGet(context);                                               \
    auto call =                                                                                    \
        llzkFunction_CallOpBuildToCallee(builder, location, f.op, 0, (const MlirValue *)NULL);     \
    if (llzkFunction_CallOpGetCalleeIsStructCompute(call)) {                                       \
      func(call);                                                                                  \
    }                                                                                              \
    mlirOperationDestroy(call);                                                                    \
    mlirOpBuilderDestroy(builder);                                                                 \
  }

link_test_call_op(
    llzk_call_op_get_self_value_from_compute, llzkFunction_CallOpGetSelfValueFromCompute
);
link_test_call_op(
    llzk_call_op_get_self_value_from_constrain, llzkFunction_CallOpGetSelfValueFromConstrain
);
link_test_call_op(
    llzk_call_op_get_single_result_type_of_compute, llzkFunction_CallOpGetSingleResultTypeOfCompute
);
