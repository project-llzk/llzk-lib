//===-- POD.cpp -------------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk-c/Dialect/POD.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/POD/IR/Attrs.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"

#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

#include <mlir-c/BuiltinAttributes.h>
#include <mlir-c/BuiltinTypes.h>
#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringRef.h>

#include <gtest/gtest.h>

#include "../CAPITestBase.h"

// Include the auto-generated tests
#include "llzk/Dialect/POD/IR/Attrs.capi.test.cpp.inc"
#include "llzk/Dialect/POD/IR/Dialect.capi.test.cpp.inc"
#include "llzk/Dialect/POD/IR/Ops.capi.test.cpp.inc"
#include "llzk/Dialect/POD/IR/Types.capi.test.cpp.inc"

namespace {

llzk::pod::RecordAttr createRecordAttrCpp(mlir::StringAttr name, mlir::Type type) {
  return llzk::pod::RecordAttr::get(type.getContext(), name, type);
}

llzk::pod::RecordAttr createRecordAttrCpp(mlir::StringRef name, mlir::Type type) {
  return createRecordAttrCpp(mlir::StringAttr::get(type.getContext(), name), type);
}

} // namespace

class PODDialectTests : public CAPITest {
protected:
  mlir::Type unwrappedIndexType() { return unwrap(createIndexType()); }

  llzk::pod::RecordAttr testRecord(mlir::StringRef name) {
    return createRecordAttrCpp(name, unwrappedIndexType());
  }

  llvm::SmallVector<MlirOperation> createNOps(int64_t n_ops, MlirType elt_type) {
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

TEST_F(PODDialectTests, llzkPod_RecordAttrGet) {
  auto name = mlirStringRefCreateFromCString("a record name");
  auto type = createIndexType();
  auto attr = llzkPod_RecordAttrGetInferredContext(mlirIdentifierGet(context, name), type);

  auto unwrapped = mlir::unwrap_cast<llzk::pod::RecordAttr>(attr);
  ASSERT_EQ(unwrapped.getName(), "a record name");
  ASSERT_EQ(unwrapped.getType(), unwrappedIndexType());
}

TEST_F(PODDialectTests, llzkPod_RecordAttrGetName) {
  auto attr = testRecord("a record name");
  auto name = llzkPod_RecordAttrGetName(wrap(attr));
  ASSERT_EQ(unwrap(name), "a record name");
}

TEST_F(PODDialectTests, llzkPod_RecordAttrGetNameSym) {
  auto attr = testRecord("a record name");
  auto name = llzkPod_RecordAttrGetNameSym(wrap(attr));
  ASSERT_EQ(
      unwrap(name),
      mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(unwrap(context), "a record name"))
  );
}

TEST_F(PODDialectTests, llzkPod_RecordAttrGetType) {
  auto attr = testRecord("a record name");
  auto type = llzkPod_RecordAttrGetType(wrap(attr));
  ASSERT_EQ(unwrap(type), unwrappedIndexType());
}

TEST_F(PODDialectTests, llzkPod_PodTypeGet) {
  auto record = testRecord("a record name");
  auto recordWrapped = wrap(record);
  auto type = llzkPod_PodTypeGet(context, 1, &recordWrapped);
  mlir::Type expected = llzk::pod::PodType::get(unwrap(context), {record});
  ASSERT_EQ(unwrap(type), expected);
}

TEST_F(PODDialectTests, llzkPod_PodTypeGetFromInitialValues) {
  auto ops = createNOps(2, createIndexType());
  LlzkRecordValue initialValues[] = {
      LlzkRecordValue {
          .name = mlirStringRefCreateFromCString("x"), .value = mlirOperationGetResult(ops[0], 0)
      },
      LlzkRecordValue {
          .name = mlirStringRefCreateFromCString("y"), .value = mlirOperationGetResult(ops[1], 0)
      },
  };
  auto type = llzkPod_PodTypeGetFromInitialValues(context, 2, initialValues);
  mlir::Type expected =
      llzk::pod::PodType::get(unwrap(context), {testRecord("x"), testRecord("y")});
  ASSERT_EQ(unwrap(type), expected);

  for (auto op : ops) {
    mlirOperationDestroy(op);
  }
}

TEST_F(PODDialectTests, llzkPod_PodTypeGetRecords) {
  auto record = wrap(testRecord("a record name"));
  auto type = llzkPod_PodTypeGet(context, 1, &record);
  MlirAttribute output[1];
  llzkPod_PodTypeGetRecords(type, output);
  ASSERT_NE(output[0].ptr, nullptr);
  ASSERT_EQ(output[0].ptr, record.ptr);
}

// Implementation for `ReadPodOp_build_pass` test
std::unique_ptr<ReadPodOpBuildFuncHelper> ReadPodOpBuildFuncHelper::get() {
  struct Impl : public ReadPodOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      mlir::StringRef name = "RecordName";
      MlirType indexTy = testClass.createIndexType();
      auto podTy = llzk::pod::PodType::get(
          unwrap(testClass.context), {createRecordAttrCpp(name, unwrap(indexTy))}
      );
      auto newPodOp = unwrap(builder)->create<llzk::pod::NewPodOp>(unwrap(location), podTy);
      auto recordName = mlir::FlatSymbolRefAttr::get(unwrap(testClass.context), name);
      return llzkPod_ReadPodOpBuild(
          builder, location, indexTy, wrap(newPodOp.getResult()), wrap(recordName)
      );
    }
  };
  return std::make_unique<Impl>();
}

// Implementation for `WritePodOp_build_pass` test
std::unique_ptr<WritePodOpBuildFuncHelper> WritePodOpBuildFuncHelper::get() {
  struct Impl : public WritePodOpBuildFuncHelper {
    mlir::OwningOpRef<mlir::ModuleOp> parentModule;
    MlirOperation
    callBuild(const CAPITest &testClass, MlirOpBuilder builder, MlirLocation location) override {
      this->parentModule = testClass.cppGenStructAndSetInsertionPoint(
          builder, location, llzk::function::FunctionKind::StructCompute
      );
      mlir::StringRef name = "RecordName";
      MlirType indexTy = testClass.createIndexType();
      auto podTy = llzk::pod::PodType::get(
          unwrap(testClass.context), {createRecordAttrCpp(name, unwrap(indexTy))}
      );
      auto newPodOp = unwrap(builder)->create<llzk::pod::NewPodOp>(unwrap(location), podTy);
      auto recordName = mlir::FlatSymbolRefAttr::get(unwrap(testClass.context), name);
      return llzkPod_WritePodOpBuild(
          builder, location, wrap(newPodOp.getResult()),
          mlirOperationGetResult(testClass.createIndexOperation(), 0), wrap(recordName)
      );
    }
  };
  return std::make_unique<Impl>();
}
