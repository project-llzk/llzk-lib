//===-- Function.cpp --------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Function.h"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Function/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/Function/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Function/IR/Ops.capi.test.cpp.inc"

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

struct FuncDialectTest : public CAPITest {
  TestFuncDefOp test_function() const {
    auto in_types = llvm::SmallVector<MlirType>({createIndexType(), createIndexType()});
    auto in_attrs = empty_arg_attrs<2>(context);
    auto out_types = llvm::SmallVector<MlirType>({createIndexType()});
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

  TestFuncDefOp test_function0() const {
    auto in_types = llvm::SmallVector<MlirType>();
    auto out_types = llvm::SmallVector<MlirType>({createIndexType()});
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

TEST_F(FuncDialectTest, llzk_func_def_op_create_with_attrs_and_arg_attrs) {
  MlirType in_types[] = {createIndexType()};
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

TEST_F(FuncDialectTest, llzk_func_def_op_has_allow_constraint_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowConstraintAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_set_allow_constraint_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowConstraintAttr(f.op));
  llzkFunction_FuncDefOpSetAllowConstraintAttr(f.op, true);
  EXPECT_TRUE(llzkFunction_FuncDefOpHasAllowConstraintAttr(f.op));
  llzkFunction_FuncDefOpSetAllowConstraintAttr(f.op, false);
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowConstraintAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_has_allow_witness_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowWitnessAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_set_allow_witness_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowWitnessAttr(f.op));
  llzkFunction_FuncDefOpSetAllowWitnessAttr(f.op, true);
  EXPECT_TRUE(llzkFunction_FuncDefOpHasAllowWitnessAttr(f.op));
  llzkFunction_FuncDefOpSetAllowWitnessAttr(f.op, false);
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowWitnessAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_has_allow_non_native_field_ops_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_set_allow_non_native_field_ops_attr) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr(f.op));
  llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(f.op, true);
  EXPECT_TRUE(llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr(f.op));
  llzkFunction_FuncDefOpSetAllowNonNativeFieldOpsAttr(f.op, false);
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasAllowNonNativeFieldOpsAttr(f.op));
}

TEST_F(FuncDialectTest, llzk_func_def_op_has_arg_is_pub) {
  auto f = test_function();
  EXPECT_TRUE(!llzkFunction_FuncDefOpHasArgPublicAttr(f.op, 0));
}

TEST_F(FuncDialectTest, llzk_func_def_op_get_fully_qualified_name) {
  // Because the func is not included in a module or struct calling this method will result
  // in an error. To avoid this while still having a test that links against the function we
  // only "call" the method on a condition that is actually impossible but the compiler
  // cannot see that.
  auto f = test_function();
  if (f.op.ptr == (void *)NULL) {
    llzkFunction_FuncDefOpGetFullyQualifiedName(f.op, true);
  }
}

#define false_pred_test(name, func)                                                                \
  TEST_F(FuncDialectTest, name) {                                                                  \
    auto f = test_function();                                                                      \
    EXPECT_FALSE(func(f.op));                                                                      \
  }

false_pred_test(llzk_func_def_op_name_is_compute, llzkFunction_FuncDefOpNameIsCompute);
false_pred_test(llzk_func_def_op_name_is_constrain, llzkFunction_FuncDefOpNameIsConstrain);
false_pred_test(llzk_func_def_op_is_in_struct, llzkFunction_FuncDefOpIsInStruct);
false_pred_test(llzk_func_def_op_is_struct_compute, llzkFunction_FuncDefOpIsStructCompute);
false_pred_test(llzk_func_def_op_is_struct_constrain, llzkFunction_FuncDefOpIsStructConstrain);

struct CallOpBuildFuncHelper : public TestAnyBuildFuncHelper<FuncDialectTest> {
  bool callIsA(MlirOperation op) override { return llzkOperationIsA_Function_CallOp(op); }
};

TEST_F(FuncDialectTest, llzk_call_op_build) {
  struct : CallOpBuildFuncHelper {
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      auto callee_name = mlirFlatSymbolRefAttrGet(testClass.context, f.nameRef());
      return llzkFunction_CallOpBuild(
          builder, location, f.out_types.size(), f.out_types.data(), callee_name, 0,
          (const MlirValue *)NULL
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(FuncDialectTest, llzk_call_op_build_to_callee) {
  struct : CallOpBuildFuncHelper {
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      return llzkFunction_CallOpBuildToCallee(builder, location, f.op, 0, (const MlirValue *)NULL);
    }
  } helper;
  helper.run(*this);
}

TEST_F(FuncDialectTest, llzk_call_op_build_with_map_operands) {
  struct : CallOpBuildFuncHelper {
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      auto callee_name = mlirFlatSymbolRefAttrGet(testClass.context, f.nameRef());
      auto dims_per_map = mlirDenseI32ArrayGet(testClass.context, 0, (const int *)NULL);
      return llzkFunction_CallOpBuildWithMapOperands(
          builder, location, f.out_types.size(), f.out_types.data(), callee_name, 0,
          (const MlirValueRange *)NULL, dims_per_map, 0, (const MlirValue *)NULL
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(FuncDialectTest, llzk_call_op_build_with_map_operands_and_dims) {
  struct : CallOpBuildFuncHelper {
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      auto callee_name = mlirFlatSymbolRefAttrGet(testClass.context, f.nameRef());
      return llzkFunction_CallOpBuildWithMapOperandsAndDims(
          builder, location, f.out_types.size(), f.out_types.data(), callee_name, 0,
          (const MlirValueRange *)NULL, 0, (const int *)NULL, 0, (const MlirValue *)NULL
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(FuncDialectTest, llzk_call_op_build_to_callee_with_map_operands) {
  struct : CallOpBuildFuncHelper {
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      auto dims_per_map = mlirDenseI32ArrayGet(testClass.context, 0, (const int *)NULL);
      return llzkFunction_CallOpBuildToCalleeWithMapOperands(
          builder, location, f.op, 0, (const MlirValueRange *)NULL, dims_per_map, 0,
          (const MlirValue *)NULL
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(FuncDialectTest, llzk_call_op_build_to_callee_with_map_operands_and_dims) {
  struct : CallOpBuildFuncHelper {
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      return llzkFunction_CallOpBuildToCalleeWithMapOperandsAndDims(
          builder, location, f.op, 0, (const MlirValueRange *)NULL, 0, (const int *)NULL, 0,
          (const MlirValue *)NULL
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(FuncDialectTest, llzk_call_op_get_callee_type) {
  struct : CallOpBuildFuncHelper {
    MlirType func_type;
    MlirOperation callBuild(
        const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      auto f = testClass.test_function0();
      this->func_type = create_func_type(testClass.context, f.in_types, f.out_types);
      return llzkFunction_CallOpBuildToCallee(builder, location, f.op, 0, (const MlirValue *)NULL);
    }
    void doOtherChecks(MlirOperation op) override {
      auto out_type = llzkFunction_CallOpGetCalleeType(op);
      EXPECT_TRUE(mlirTypeEqual(this->func_type, out_type));
    }
  } helper;
  helper.run(*this);
}

#define call_pred_test(name, func, expected)                                                       \
  TEST_F(FuncDialectTest, name) {                                                                  \
    struct : CallOpBuildFuncHelper {                                                               \
      MlirOperation callBuild(                                                                     \
          const FuncDialectTest &testClass, MlirOpBuilder builder, MlirLocation location           \
      ) override {                                                                                 \
        auto f = testClass.test_function0();                                                       \
        return llzkFunction_CallOpBuildToCallee(                                                   \
            builder, location, f.op, 0, (const MlirValue *)NULL                                    \
        );                                                                                         \
      }                                                                                            \
      void doOtherChecks(MlirOperation op) override { EXPECT_EQ(func(op), expected); }             \
    } helper;                                                                                      \
    helper.run(*this);                                                                             \
  }

call_pred_test(test_llzk_operation_is_a_call_op_pass, llzkOperationIsA_Function_CallOp, true);
call_pred_test(test_llzk_call_op_callee_is_compute, llzkFunction_CallOpCalleeIsCompute, false);
call_pred_test(test_llzk_call_op_callee_is_constrain, llzkFunction_CallOpCalleeIsConstrain, false);
call_pred_test(
    test_llzk_call_op_callee_is_struct_compute, llzkFunction_CallOpCalleeIsStructCompute, false
);
call_pred_test(
    test_llzk_call_op_callee_is_struct_constrain, llzkFunction_CallOpCalleeIsStructConstrain, false
);

// Implementation for `ReturnOp_build_pass` test
std::unique_ptr<ReturnOpBuildFuncHelper> ReturnOpBuildFuncHelper::get() {
  struct Impl : public ReturnOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
        llzk::ModuleBuilder cppBldr(this->parentModule.get());
        auto name = "func_name";
        cppBldr.insertFreeFunc(
            name,
            mlir::FunctionType::get(
                unwrap(testClass.context), mlir::TypeRange {}, mlir::TypeRange {}
            ),
            unwrap(location)
        );
        llzk::function::FuncDefOp fDef = cppBldr.getFreeFunc(name).value();
        mlirOpBuilderSetInsertionPointToStart(builder, wrap(&fDef.getBody().emplaceBlock()));
      }
      llvm::SmallVector<MlirValue> vals {};
      return llzkFunction_ReturnOpBuild(builder, location, vals.size(), vals.data());
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `FuncDefOp_build_pass` test
std::unique_ptr<FuncDefOpBuildFuncHelper> FuncDefOpBuildFuncHelper::get() {
  struct Impl : public FuncDefOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      MlirContext ctx = testClass.context;
      mlir::FunctionType fTy;
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        // Use ModuleOp as parent to avoid the following:
        // error: 'function.def' op expects parent op to be one of 'builtin.module, struct.def'
        this->parentModule = testClass.cppNewModuleAndSetInsertionPoint(builder, location);
        // setup function type
        fTy = mlir::FunctionType::get(unwrap(ctx), mlir::TypeRange {}, mlir::TypeRange {});
      }
      auto result = llzkFunction_FuncDefOpBuild(
          builder, location, mlirIdentifierGet(ctx, mlirStringRefCreateFromCString("funcName")),
          mlirTypeAttrGet(wrap(fTy)), mlirArrayAttrGet(ctx, 0, NULL), mlirArrayAttrGet(ctx, 0, NULL)
      );
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      {
        // Add 'private' visibility to avoid the following (since function has no body):
        // error: 'function.def' op symbol declaration cannot have public visibility
        mlir::unwrap_cast<llzk::function::FuncDefOp>(result).setVisibility(
            mlir::SymbolTable::Visibility::Private
        );
      }
      return result;
    }
  };
  return std::make_unique<Impl>();
}
