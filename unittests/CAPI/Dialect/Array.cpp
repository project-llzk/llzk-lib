//===-- Array.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"

#include "llzk-c/Dialect/Array.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/Array/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Array/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Array/IR/Types.capi.test.cpp.inc"

struct ArrayDialectTests : public CAPITest {
  MlirType test_array(MlirType elt, llvm::ArrayRef<int64_t> dims) const {
    return llzkArray_ArrayTypeGetWithShape(elt, dims.size(), dims.data());
  }

  llvm::SmallVector<MlirOperation> create_n_ops(int64_t n_ops, MlirType elt_type) const {
    auto name = mlirStringRefCreateFromCString("arith.constant");
    auto attr_name = mlirIdentifierGet(context, mlirStringRefCreateFromCString("value"));
    auto location = mlirLocationUnknownGet(context);
    llvm::SmallVector<MlirOperation> result;
    for (int64_t n = 0; n < n_ops; n++) {
      auto attr = mlirNamedAttributeGet(attr_name, mlirIntegerAttrGet(elt_type, n));
      auto op_state = mlirOperationStateGet(name, location);
      mlirOperationStateAddResults(&op_state, 1, &elt_type);
      mlirOperationStateAddAttributes(&op_state, 1, &attr);

      auto created_op = mlirOperationCreate(&op_state);

      result.push_back(created_op);
    }
    return result;
  }
};

TEST_F(ArrayDialectTests, array_type_get) {
  auto size = createIndexAttribute(1);
  MlirAttribute dims[1] = {size};
  auto arr_type = llzkArray_ArrayTypeGetWithDims(createIndexType(), 1, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
}

TEST_F(ArrayDialectTests, type_is_a_array_type_pass) {
  auto size = createIndexAttribute(1);
  MlirAttribute dims[1] = {size};
  auto arr_type = llzkArray_ArrayTypeGetWithDims(createIndexType(), 1, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  EXPECT_TRUE(llzkTypeIsA_Array_ArrayType(arr_type));
}

TEST_F(ArrayDialectTests, array_type_get_with_numeric_dims) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArray_ArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
}

TEST_F(ArrayDialectTests, array_type_get_element_type) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArray_ArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto elt_type = llzkArray_ArrayTypeGetElementType(arr_type);
  EXPECT_TRUE(mlirTypeEqual(createIndexType(), elt_type));
}

TEST_F(ArrayDialectTests, array_type_get_num_dims) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArray_ArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto n_dims = llzkArray_ArrayTypeGetDimensionSizesCount(arr_type);
  EXPECT_EQ(n_dims, 2);
}

TEST_F(ArrayDialectTests, array_type_get_dim) {
  int64_t dims[2] = {1, 2};
  auto arr_type = llzkArray_ArrayTypeGetWithShape(createIndexType(), 2, dims);
  EXPECT_NE(arr_type.ptr, (const void *)NULL);
  auto out_dim = llzkArray_ArrayTypeGetDimensionSizesAt(arr_type, 0);
  auto dim_as_attr = createIndexAttribute(dims[0]);
  EXPECT_TRUE(mlirAttributeEqual(out_dim, dim_as_attr));
}

struct CreateArrayOpBuildFuncHelper : public TestAnyBuildFuncHelper<ArrayDialectTests> {
  bool callIsA(MlirOperation op) override { return llzkOperationIsA_Array_CreateArrayOp(op); }
};

TEST_F(ArrayDialectTests, create_array_op_build_with_values) {
  struct LocalHelper : CreateArrayOpBuildFuncHelper {
    llvm::SmallVector<MlirOperation> otherOps;

    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      this->otherOps = testClass.create_n_ops(1, elt_type);
      llvm::SmallVector<MlirValue> values;
      for (auto op : this->otherOps) {
        values.push_back(mlirOperationGetResult(op, 0));
      }
      return llzkArray_CreateArrayOpBuildWithValues(
          builder, location, test_type, values.size(), values.data()
      );
    }
    void doOtherChecks(MlirOperation) override {
      for (auto op : this->otherOps) {
        EXPECT_TRUE(mlirOperationVerify(op));
      }
    }
    ~LocalHelper() override {
      for (auto op : this->otherOps) {
        mlirOperationDestroy(op);
      }
    }
  } helper;
  helper.run(*this);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands) {
  struct : CreateArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      auto dims_per_map = mlirDenseI32ArrayGet(testClass.context, 0, NULL);
      return llzkArray_CreateArrayOpBuildWithMapOperands(
          builder, location, test_type, 0, NULL, dims_per_map
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands_and_dims) {
  struct : CreateArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      return llzkArray_CreateArrayOpBuildWithMapOperandsAndDims(
          builder, location, test_type, 0, NULL, 0, NULL
      );
    }
  } helper;
  helper.run(*this);
}

// Implementation for `ArrayLengthOp_build_pass` test
std::unique_ptr<ArrayLengthOpBuildFuncHelper> ArrayLengthOpBuildFuncHelper::get() {
  struct Impl : public ArrayLengthOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Value array;
      mlir::Value dim;
      {
        mlir::Location cppLoc = unwrap(location);
        mlir::OpBuilder *bldr = unwrap(builder);
        auto idxType = bldr->getIndexType();
        auto intAttr1 = bldr->getIntegerAttr(idxType, 1);
        array = bldr->create<llzk::array::CreateArrayOp>(
            cppLoc, llzk::array::ArrayType::get(idxType, llvm::ArrayRef<mlir::Attribute> {intAttr1})
        );
        dim = bldr->create<mlir::arith::ConstantOp>(cppLoc, idxType, intAttr1);
      }
      return llzkArray_ArrayLengthOpBuild(builder, location, wrap(array), wrap(dim));
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ReadArrayOp_build_pass` test
std::unique_ptr<ReadArrayOpBuildFuncHelper> ReadArrayOpBuildFuncHelper::get() {
  struct Impl : public ReadArrayOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Type elemType;
      mlir::Value array;
      llvm::SmallVector<MlirValue> indices;
      {
        mlir::Location cppLoc = unwrap(location);
        mlir::OpBuilder *bldr = unwrap(builder);
        auto idxType = bldr->getIndexType();
        auto intAttr0 = bldr->getIntegerAttr(idxType, 0);
        elemType = idxType;
        array = bldr->create<llzk::array::CreateArrayOp>(
            cppLoc, llzk::array::ArrayType::get(idxType, llvm::ArrayRef<mlir::Attribute> {intAttr0})
        );
        mlir::Value idx = bldr->create<mlir::arith::ConstantOp>(cppLoc, idxType, intAttr0);
        indices.push_back(wrap(idx));
      }
      return llzkArray_ReadArrayOpBuild(
          builder, location, wrap(elemType), wrap(array), indices.size(), indices.data()
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `WriteArrayOp_build_pass` test
std::unique_ptr<WriteArrayOpBuildFuncHelper> WriteArrayOpBuildFuncHelper::get() {
  struct Impl : public WriteArrayOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Value array;
      mlir::Value value;
      llvm::SmallVector<MlirValue> indices;
      {
        mlir::Location cppLoc = unwrap(location);
        mlir::OpBuilder *bldr = unwrap(builder);
        auto idxType = bldr->getIndexType();
        auto intAttr0 = bldr->getIntegerAttr(idxType, 0);
        array = bldr->create<llzk::array::CreateArrayOp>(
            cppLoc, llzk::array::ArrayType::get(idxType, llvm::ArrayRef<mlir::Attribute> {intAttr0})
        );
        mlir::Value idx = bldr->create<mlir::arith::ConstantOp>(cppLoc, idxType, intAttr0);
        indices.push_back(wrap(idx));
        value = idx;
      }
      return llzkArray_WriteArrayOpBuild(
          builder, location, wrap(array), indices.size(), indices.data(), wrap(value)
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `InsertArrayOp_build_pass` test
std::unique_ptr<InsertArrayOpBuildFuncHelper> InsertArrayOpBuildFuncHelper::get() {
  struct Impl : public InsertArrayOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Value array_big;
      mlir::Value array_small;
      llvm::SmallVector<MlirValue> indices;
      {
        mlir::Location cppLoc = unwrap(location);
        mlir::OpBuilder *bldr = unwrap(builder);
        auto idxType = bldr->getIndexType();
        auto intAttr0 = bldr->getIntegerAttr(idxType, 0);
        auto intAttr1 = bldr->getIntegerAttr(idxType, 1);
        array_big = bldr->create<llzk::array::CreateArrayOp>(
            cppLoc, llzk::array::ArrayType::get(
                        idxType, llvm::ArrayRef<mlir::Attribute> {intAttr1, intAttr1}
                    )
        );
        mlir::Value idx = bldr->create<mlir::arith::ConstantOp>(cppLoc, idxType, intAttr0);
        indices.push_back(wrap(idx));
        array_small = bldr->create<llzk::array::CreateArrayOp>(
            cppLoc, llzk::array::ArrayType::get(idxType, llvm::ArrayRef<mlir::Attribute> {intAttr1})
        );
      }
      return llzkArray_InsertArrayOpBuild(
          builder, location, wrap(array_big), indices.size(), indices.data(), wrap(array_small)
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `ExtractArrayOp_build_pass` test
std::unique_ptr<ExtractArrayOpBuildFuncHelper> ExtractArrayOpBuildFuncHelper::get() {
  struct Impl : public ExtractArrayOpBuildFuncHelper {
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      // Use C++ API to avoid indirectly testing other LLZK C API functions here.
      mlir::Value array_big;
      mlir::Type small_type;
      llvm::SmallVector<MlirValue> indices;
      {
        mlir::Location cppLoc = unwrap(location);
        mlir::OpBuilder *bldr = unwrap(builder);
        auto idxType = bldr->getIndexType();
        auto intAttr0 = bldr->getIntegerAttr(idxType, 0);
        auto intAttr1 = bldr->getIntegerAttr(idxType, 1);
        array_big = bldr->create<llzk::array::CreateArrayOp>(
            cppLoc, llzk::array::ArrayType::get(
                        idxType, llvm::ArrayRef<mlir::Attribute> {intAttr1, intAttr1}
                    )
        );
        mlir::Value idx = bldr->create<mlir::arith::ConstantOp>(cppLoc, idxType, intAttr0);
        indices.push_back(wrap(idx));
        small_type =
            llzk::array::ArrayType::get(idxType, llvm::ArrayRef<mlir::Attribute> {intAttr1});
      }
      return llzkArray_ExtractArrayOpBuild(
          builder, location, wrap(small_type), wrap(array_big), indices.size(), indices.data()
      );
    }
  };
  return std::make_unique<Impl>();
}
