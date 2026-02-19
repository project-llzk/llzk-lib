//===-- POD.cpp - POD dialect C API implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"
#include "llzk/CAPI/Support.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"

#include "llzk-c/Dialect/POD.h"
#include "llzk-c/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Registration.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/Support/LLVM.h>

#include <mlir-c/IR.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVectorExtras.h>

#include <cstdint>

using namespace mlir;
using namespace llzk;
using namespace llzk::pod;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(POD, llzk__pod, PODDialect)

namespace {

static SmallVector<RecordValue>
fromRawRecordValues(intptr_t nValues, LlzkRecordValue const *values) {
  return llvm::map_to_vector(ArrayRef(values, nValues), [](const auto &record) {
    return RecordValue {.name = unwrap(record.name), .value = unwrap(record.value)};
  });
}

} // namespace

//===----------------------------------------------------------------------===//
// RecordAttr
//===----------------------------------------------------------------------===//

MlirAttribute llzkRecordAttrGet(MlirStringRef name, MlirType type) {
  auto t = unwrap(type);
  return wrap(RecordAttr::get(t.getContext(), StringAttr::get(t.getContext(), unwrap(name)), t));
}

bool llzkAttributeIsARecordAttr(MlirAttribute attr) { return mlir::isa<RecordAttr>(unwrap(attr)); }

MlirStringRef llzkRecordAttrGetName(MlirAttribute attr) {
  return wrap(unwrap_cast<RecordAttr>(attr).getName().getValue());
}

MlirAttribute llzkRecordAttrGetNameSym(MlirAttribute attr) {
  return wrap(unwrap_cast<RecordAttr>(attr).getNameSym());
}

MlirType llzkRecordAttrGetType(MlirAttribute attr) {
  return wrap(unwrap_cast<RecordAttr>(attr).getType());
}

//===----------------------------------------------------------------------===//
// PodType
//===----------------------------------------------------------------------===//

MlirType llzkPodTypeGet(MlirContext context, intptr_t nRecords, MlirAttribute const *records) {
  SmallVector<Attribute> recordsSto;
  auto recordAttrs = llvm::map_to_vector(unwrapList(nRecords, records, recordsSto), [](auto attr) {
    return mlir::cast<RecordAttr>(attr);
  });
  return wrap(PodType::get(unwrap(context), recordAttrs));
}

MlirType llzkPodTypeGetFromInitialValues(
    MlirContext context, intptr_t nRecords, LlzkRecordValue const *records
) {
  auto initialValues = fromRawRecordValues(nRecords, records);
  return wrap(PodType::fromInitialValues(unwrap(context), initialValues));
}

bool llzkTypeIsAPodType(MlirType type) { return mlir::isa<PodType>(unwrap(type)); }

intptr_t llzkPodTypeGetNumRecords(MlirType type) {
  return static_cast<intptr_t>(unwrap_cast<PodType>(type).getRecords().size());
}

void llzkPodTypeGetRecords(MlirType type, MlirAttribute *dst) {
  auto records = unwrap_cast<PodType>(type).getRecords();
  MutableArrayRef<MlirAttribute> dstRef(dst, records.size());
  llvm::transform(records, dstRef.begin(), [](auto record) { return wrap(record); });
}

MlirAttribute llzkPodTypeGetNthRecord(MlirType type, intptr_t n) {
  return wrap(unwrap_cast<PodType>(type).getRecords()[n]);
}

namespace {
static MlirType
lookupRecordImpl(PodType type, StringRef name, llvm::function_ref<InFlightDiagnostic()> emitError) {
  auto attr = type.getRecord(name, emitError);
  if (failed(attr)) {
    return MlirType {.ptr = nullptr};
  }
  return wrap(*attr);
}
} // namespace

MlirType llzkPodTypeLookupRecord(MlirType type, MlirStringRef name) {
  auto pod = unwrap_cast<PodType>(type);
  return lookupRecordImpl(pod, unwrap(name), [pod] {
    auto *ctx = pod.getContext();
    return ctx->getDiagEngine().emit(Builder(ctx).getUnknownLoc(), DiagnosticSeverity::Error);
  });
}

MlirType
llzkPodTypeLookupRecordWithinLocation(MlirType type, MlirStringRef name, MlirLocation loc) {
  auto pod = unwrap_cast<PodType>(type);
  return lookupRecordImpl(pod, unwrap(name), [pod, loc] {
    return pod.getContext()->getDiagEngine().emit(unwrap(loc), DiagnosticSeverity::Error);
  });
}

MlirType
llzkPodTypeLookupRecordWithinOperation(MlirType type, MlirStringRef name, MlirOperation op) {
  return lookupRecordImpl(unwrap_cast<PodType>(type), unwrap(name), [op] {
    return unwrap(op)->emitError();
  });
}

//===----------------------------------------------------------------------===//
// NewPodOp
//===----------------------------------------------------------------------===//

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Pod, NewPodOp, InferredFromInitialValues, intptr_t nValues, LlzkRecordValue const *values
) {
  auto recordValues = fromRawRecordValues(nValues, values);
  return wrap(create<NewPodOp>(builder, location, recordValues));
}

LLZK_DEFINE_OP_BUILD_METHOD(
    Pod, NewPodOp, MlirType type, intptr_t nValues, LlzkRecordValue const *values
) {
  auto recordValues = fromRawRecordValues(nValues, values);
  return wrap(create<NewPodOp>(builder, location, unwrap_cast<PodType>(type), recordValues));
}

LLZK_DEFINE_SUFFIX_OP_BUILD_METHOD(
    Pod, NewPodOp, WithMapOperands, MlirType type, intptr_t nValues, LlzkRecordValue const *values,
    LlzkAffineMapOperandsBuilder mapOperands
) {
  auto recordValues = fromRawRecordValues(nValues, values);
  MapOperandsHelper<> mapOps(mapOperands.nMapOperands, mapOperands.mapOperands);
  auto numDimsPerMap =
      llzkAffineMapOperandsBuilderGetDimsPerMapAttr(mapOperands, mlirLocationGetContext(location));
  return wrap(
      create<NewPodOp>(
          builder, location, unwrap_cast<PodType>(type), *mapOps,
          unwrap_cast<DenseI32ArrayAttr>(numDimsPerMap)
      )
  );
}

bool llzkOperationIsANewPodOp(MlirOperation op) { return mlir::isa<NewPodOp>(unwrap(op)); }

//===----------------------------------------------------------------------===//
// ReadPodOp
//===----------------------------------------------------------------------===//

bool llzkOperationIsAReadPodOp(MlirOperation op) { return mlir::isa<ReadPodOp>(unwrap(op)); }

//===----------------------------------------------------------------------===//
// WritePodOp
//===----------------------------------------------------------------------===//

bool llzkOperationIsAWritePodOp(MlirOperation op) { return mlir::isa<WritePodOp>(unwrap(op)); }
