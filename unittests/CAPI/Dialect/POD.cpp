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
#include "llzk/Dialect/POD/IR/Types.h"

#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/IR/Attributes.h>
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

class PODDialectTests : public CAPITest {
protected:
  mlir::Type unwrappedIndexType() { return mlir::IndexType::get(unwrap(context)); }

  llzk::pod::RecordAttr recordAttr(llvm::StringRef name, mlir::Type type) {
    return llzk::pod::RecordAttr::get(
        type.getContext(), mlir::StringAttr::get(type.getContext(), name), type
    );
  }

  MlirType testPod(llvm::ArrayRef<std::pair<MlirStringRef, MlirType>> records) {
    auto recordAttrs = llvm::map_to_vector(records, [](auto record) {
      auto [name, type] = record;
      return llzkPod_RecordAttrGet(name, type);
    });
    return llzkPod_PodTypeGet(context, static_cast<intptr_t>(records.size()), recordAttrs.data());
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
  auto attr = llzkPod_RecordAttrGet(name, type);

  auto unwrapped = mlir::unwrap_cast<llzk::pod::RecordAttr>(attr);
  ASSERT_EQ(unwrapped.getName(), "a record name");
  ASSERT_EQ(unwrapped.getType(), unwrappedIndexType());
}

TEST_F(PODDialectTests, llzkPod_RecordAttrGetName) {
  auto attr = recordAttr("a record name", unwrappedIndexType());
  auto name = llzkPod_RecordAttrGetName(wrap(attr));
  ASSERT_EQ(unwrap(name), "a record name");
}

TEST_F(PODDialectTests, llzkPod_RecordAttrGetNameSym) {
  auto attr = recordAttr("a record name", unwrappedIndexType());
  auto name = llzkPod_RecordAttrGetNameSym(wrap(attr));
  ASSERT_EQ(
      unwrap(name),
      mlir::FlatSymbolRefAttr::get(mlir::StringAttr::get(unwrap(context), "a record name"))
  );
}

TEST_F(PODDialectTests, llzkPod_RecordAttrGetType) {
  auto attr = recordAttr("a record name", unwrappedIndexType());
  auto type = llzkPod_RecordAttrGetType(wrap(attr));
  ASSERT_EQ(unwrap(type), unwrappedIndexType());
}

TEST_F(PODDialectTests, llzkPod_PodTypeGet) {
  auto record =
      llzkPod_RecordAttrGet(mlirStringRefCreateFromCString("record name"), createIndexType());
  auto type = llzkPod_PodTypeGet(context, 1, &record);
  mlir::Type expected =
      llzk::pod::PodType::get(unwrap(context), {recordAttr("record name", unwrappedIndexType())});
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
  mlir::Type expected = llzk::pod::PodType::get(
      unwrap(context),
      {recordAttr("x", unwrappedIndexType()), recordAttr("y", unwrappedIndexType())}
  );
  ASSERT_EQ(unwrap(type), expected);

  for (auto op : ops) {
    mlirOperationDestroy(op);
  }
}

TEST_F(PODDialectTests, llzkPod_PodTypeGetRecords) {
  auto record =
      llzkPod_RecordAttrGet(mlirStringRefCreateFromCString("record_name"), createIndexType());
  auto type = llzkPod_PodTypeGet(context, 1, &record);
  MlirAttribute output[1];
  llzkPod_PodTypeGetRecords(type, output);
  ASSERT_NE(output[0].ptr, nullptr);
  ASSERT_EQ(output[0].ptr, record.ptr);
}
