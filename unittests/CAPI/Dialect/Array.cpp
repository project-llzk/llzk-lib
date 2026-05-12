//===-- Array.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/Array.h"

#include "../CAPITestBase.h"

#include "llzk-c/Support.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Util/Compare.h"

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>

// Include the auto-generated tests
#include "llzk/Dialect/Array/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/Array/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/Array/IR/Types.capi.test.cpp.inc"

struct ArrayDialectTests : public CAPITest {
  MlirType test_array(MlirType elt, llvm::ArrayRef<int64_t> dims) const {
    return llzkArray_ArrayTypeGetWithShape(
        elt, llzk::checkedCast<intptr_t>(dims.size()), dims.data()
    );
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
          builder, location, test_type, llzk::checkedCast<intptr_t>(values.size()), values.data()
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
  struct LocalHelper : CreateArrayOpBuildFuncHelper {
    LlzkAffineMapOperandsBuilder affineOperandsBuilder;
    LocalHelper() { affineOperandsBuilder = llzkAffineMapOperandsBuilderCreate(); }
    ~LocalHelper() override { llzkAffineMapOperandsBuilderDestroy(&affineOperandsBuilder); }

    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      affineOperandsBuilder.nDimsPerMap = -1;
      affineOperandsBuilder.dimsPerMap.attr = mlirDenseI32ArrayGet(testClass.context, 0, NULL);
      return llzkArray_CreateArrayOpBuildWithMapOperands(
          builder, location, test_type, affineOperandsBuilder
      );
    }
  } helper;
  helper.run(*this);
}

TEST_F(ArrayDialectTests, create_array_op_build_with_map_operands_and_dims) {
  struct LocalHelper : CreateArrayOpBuildFuncHelper {
    LlzkAffineMapOperandsBuilder affineOperandsBuilder;
    LocalHelper() { affineOperandsBuilder = llzkAffineMapOperandsBuilderCreate(); }
    ~LocalHelper() override { llzkAffineMapOperandsBuilderDestroy(&affineOperandsBuilder); }

    MlirOperation callBuild(
        const ArrayDialectTests &testClass, MlirOpBuilder builder, MlirLocation location
    ) override {
      int64_t dims[1] = {1};
      auto elt_type = testClass.createIndexType();
      auto test_type = testClass.test_array(elt_type, llvm::ArrayRef(dims, 1));
      return llzkArray_CreateArrayOpBuildWithMapOperands(
          builder, location, test_type, affineOperandsBuilder
      );
    }
  } helper;
  helper.run(*this);
}

// Implementation for `ArrayLengthOp_build_pass` test
std::unique_ptr<ArrayLengthOpBuildFuncHelper> ArrayLengthOpBuildFuncHelper::get() {
  struct Impl : public ArrayLengthOpBuildFuncHelper {
    MlirOperation callBuild(
        const CAPITest & /*testClass*/, MlirOpBuilder builder, MlirLocation location
    ) override {
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
    MlirOperation callBuild(
        const CAPITest & /*testClass*/, MlirOpBuilder builder, MlirLocation location
    ) override {
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
          builder, location, wrap(elemType), wrap(array),
          llzk::checkedCast<intptr_t>(indices.size()), indices.data()
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `WriteArrayOp_build_pass` test
std::unique_ptr<WriteArrayOpBuildFuncHelper> WriteArrayOpBuildFuncHelper::get() {
  struct Impl : public WriteArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const CAPITest & /*testClass*/, MlirOpBuilder builder, MlirLocation location
    ) override {
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
          builder, location, wrap(array), llzk::checkedCast<intptr_t>(indices.size()),
          indices.data(), wrap(value)
      );
    }
  };
  return std::make_unique<Impl>();
}

/// Regression test for ops with a fixed operand after a variadic operand segment.
///
/// `array.write` has operands laid out as:
///   [arr_ref, indices..., rvalue]
/// The C API accessors must use MLIR's generated ODS segment index/length logic so that `rvalue`
/// is found after however many `indices` operands are present, not at the static ODS index 2.
TEST_F(ArrayDialectTests, write_array_op_accessors_handle_fixed_operand_after_variadic_indices) {
  MlirOpBuilder builder = mlirOpBuilderCreate(context);
  MlirLocation location = mlirLocationUnknownGet(context);
  auto eltType = createIndexType();

  int64_t dims[2] = {1, 1};
  auto arrType = test_array(eltType, llvm::ArrayRef(dims, 2));
  auto arrState = mlirOperationStateGet(mlirStringRefCreateFromCString("array.new"), location);
  mlirOperationStateAddResults(&arrState, 1, &arrType);
  MlirOperation arrOp = mlirOperationCreate(&arrState);
  ASSERT_NE(arrOp.ptr, nullptr);

  auto auxOps = create_n_ops(5, eltType);
  MlirValue indices[2] = {
      mlirOperationGetResult(auxOps[0], 0),
      mlirOperationGetResult(auxOps[1], 0),
  };
  MlirValue rvalue = mlirOperationGetResult(auxOps[2], 0);

  MlirOperation op = llzkArray_WriteArrayOpBuild(
      builder, location, mlirOperationGetResult(arrOp, 0), 2, indices, rvalue
  );
  ASSERT_NE(op.ptr, nullptr);

  EXPECT_EQ(mlirOperationGetNumOperands(op), 4);
  EXPECT_EQ(llzkArray_WriteArrayOpGetIndicesCount(op), 2);
  EXPECT_EQ(llzkArray_WriteArrayOpGetIndicesAt(op, 0).ptr, indices[0].ptr);
  EXPECT_EQ(llzkArray_WriteArrayOpGetIndicesAt(op, 1).ptr, indices[1].ptr);
  EXPECT_EQ(llzkArray_WriteArrayOpGetRvalue(op).ptr, rvalue.ptr);

  // Updating the fixed trailing operand should touch the physical operand after the variadic
  // segment, not the static ODS index.
  MlirValue newRvalue = mlirOperationGetResult(auxOps[4], 0);
  llzkArray_WriteArrayOpSetRvalue(op, newRvalue);
  EXPECT_EQ(llzkArray_WriteArrayOpGetIndicesAt(op, 1).ptr, indices[1].ptr);
  EXPECT_EQ(llzkArray_WriteArrayOpGetRvalue(op).ptr, newRvalue.ptr);
  EXPECT_EQ(mlirOperationGetOperand(op, 3).ptr, newRvalue.ptr);

  // Resizing the variadic segment should keep the trailing fixed operand accessible at its new
  // physical position.
  MlirValue newIndices[1] = {mlirOperationGetResult(auxOps[3], 0)};
  llzkArray_WriteArrayOpSetIndices(op, 1, newIndices);
  EXPECT_EQ(mlirOperationGetNumOperands(op), 3);
  EXPECT_EQ(llzkArray_WriteArrayOpGetIndicesCount(op), 1);
  EXPECT_EQ(llzkArray_WriteArrayOpGetIndicesAt(op, 0).ptr, newIndices[0].ptr);
  EXPECT_EQ(llzkArray_WriteArrayOpGetRvalue(op).ptr, newRvalue.ptr);

  mlirOperationDestroy(op);
  mlirOperationDestroy(arrOp);
  for (auto auxOp : auxOps) {
    mlirOperationDestroy(auxOp);
  }
  mlirOpBuilderDestroy(builder);
}

// Implementation for `InsertArrayOp_build_pass` test
std::unique_ptr<InsertArrayOpBuildFuncHelper> InsertArrayOpBuildFuncHelper::get() {
  struct Impl : public InsertArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const CAPITest & /*testClass*/, MlirOpBuilder builder, MlirLocation location
    ) override {
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
          builder, location, wrap(array_big), llzk::checkedCast<intptr_t>(indices.size()),
          indices.data(), wrap(array_small)
      );
    }
  };
  return std::make_unique<Impl>();
}

/// Test that SetElements (a variadic operand setter using the dynamic operandSegmentSizes path)
/// correctly updates the `operandSegmentSizes` attribute so that a subsequent read of the
/// *other* variadic operand (mapOperands) still returns the correct value.
///
/// Before the fix, SetElements would update the operand list but leave `operandSegmentSizes`
/// stale (still [2, 1]). GetMapOperandsAt would then compute startIdx=2 from the stale
/// attribute and access operand[2], which no longer exists, returning the wrong value.
/// After the fix, `operandSegmentSizes` is updated to [1, 1], so startIdx=1 is correct.
TEST_F(ArrayDialectTests, create_array_op_set_elements_updates_operand_segment_sizes) {
  auto location = mlirLocationUnknownGet(context);
  auto elt_type = createIndexType();

  // Create three index-typed constant ops to serve as operands: e0, e1, m0
  auto auxOps = create_n_ops(3, elt_type);
  MlirValue e0 = mlirOperationGetResult(auxOps[0], 0);
  MlirValue e1 = mlirOperationGetResult(auxOps[1], 0);
  MlirValue m0 = mlirOperationGetResult(auxOps[2], 0);

  // Manually build an `array.new` op with:
  //   operands:            [e0, e1, m0]
  //   operandSegmentSizes: [2, 1]  (elements=2, mapOperands=1)
  //   mapOpGroupSizes:     [1]
  //   numDimsPerMap:       [0]
  //   result:              !array.type<2 x index>
  // mlirOperationCreate does not run the verifier, so the op does not need to be
  // semantically valid - we just need the right operand / attribute layout.
  int64_t dims[1] = {2};
  auto arr_type = test_array(elt_type, llvm::ArrayRef(dims, 1));

  auto op_state = mlirOperationStateGet(mlirStringRefCreateFromCString("array.new"), location);

  MlirValue operands[3] = {e0, e1, m0};
  mlirOperationStateAddOperands(&op_state, 3, operands);
  mlirOperationStateAddResults(&op_state, 1, &arr_type);

  int32_t segSizes[2] = {2, 1};
  int32_t groupSizes[1] = {1};
  int32_t numDims[1] = {0};
  MlirNamedAttribute attrs[3] = {
      mlirNamedAttributeGet(
          mlirIdentifierGet(context, mlirStringRefCreateFromCString("operandSegmentSizes")),
          mlirDenseI32ArrayGet(context, 2, segSizes)
      ),
      mlirNamedAttributeGet(
          mlirIdentifierGet(context, mlirStringRefCreateFromCString("mapOpGroupSizes")),
          mlirDenseI32ArrayGet(context, 1, groupSizes)
      ),
      mlirNamedAttributeGet(
          mlirIdentifierGet(context, mlirStringRefCreateFromCString("numDimsPerMap")),
          mlirDenseI32ArrayGet(context, 1, numDims)
      ),
  };
  mlirOperationStateAddAttributes(&op_state, 3, attrs);

  auto op = mlirOperationCreate(&op_state);
  ASSERT_NE(op.ptr, nullptr);

  // Verify the initial operand layout is as expected.
  EXPECT_EQ(llzkArray_CreateArrayOpGetElementsCount(op), 2);
  EXPECT_EQ(llzkArray_CreateArrayOpGetMapOperandsCount(op), 1);
  EXPECT_EQ(llzkArray_CreateArrayOpGetMapOperandsAt(op, 0).ptr, m0.ptr);

  // Update elements from [e0, e1] to [e0] only.
  // This setter must also update operandSegmentSizes from [2,1] to [1,1].
  MlirValue newElements[1] = {e0};
  llzkArray_CreateArrayOpSetElements(op, 1, newElements);

  // mapOperands was not touched, so it should still report count=1 and return m0.
  // Before the fix, operandSegmentSizes was not updated (still [2,1]), so
  // GetMapOperandsAt would compute startIdx=2 and access the now-nonexistent operand[2].
  EXPECT_EQ(llzkArray_CreateArrayOpGetMapOperandsCount(op), 1);
  EXPECT_EQ(llzkArray_CreateArrayOpGetMapOperandsAt(op, 0).ptr, m0.ptr);

  mlirOperationDestroy(op);
  for (auto auxOp : auxOps) {
    mlirOperationDestroy(auxOp);
  }
}

// Implementation for `ExtractArrayOp_build_pass` test
std::unique_ptr<ExtractArrayOpBuildFuncHelper> ExtractArrayOpBuildFuncHelper::get() {
  struct Impl : public ExtractArrayOpBuildFuncHelper {
    MlirOperation callBuild(
        const CAPITest & /*testClass*/, MlirOpBuilder builder, MlirLocation location
    ) override {
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
          builder, location, wrap(small_type), wrap(array_big),
          llzk::checkedCast<intptr_t>(indices.size()), indices.data()
      );
    }
  };
  return std::make_unique<Impl>();
}
