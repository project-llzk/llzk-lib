//===-- Builder.cpp - C API for op builder ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/CAPI/Builder.h"

#include "llzk-c/Builder.h"

#include "llzk/CAPI/Support.h"

#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Support.h>
#include <mlir/CAPI/Wrap.h>
#include <mlir/IR/Builders.h>

using namespace mlir;

using OpBuilderT = OpBuilder;

namespace {

class ListenerT : public OpBuilder::Listener {
public:
  ListenerT(MlirNotifyOperationInserted op, MlirNotifyBlockInserted block, void *data)
      : opInsertedCb(op), blockInsertedCb(block), userData(data) {}

  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint previous) final {
    MlirOpBuilderInsertPoint i {
        .block = wrap(previous.getBlock()), .point = wrap(&*previous.getPoint())
    };
    opInsertedCb(wrap(op), i, userData);
  }

  void notifyBlockInserted(Block *block, Region *previous, Region::iterator previousIt) final {
    blockInsertedCb(wrap(block), wrap(previous), wrap(&*previousIt), userData);
  }

private:
  MlirNotifyOperationInserted opInsertedCb;
  MlirNotifyBlockInserted blockInsertedCb;
  void *userData = nullptr;
};

} // namespace

//===----------------------------------------------------------------------===//
// MlirOpBuilder
//===----------------------------------------------------------------------===//

MlirOpBuilder mlirOpBuilderCreate(MlirContext ctx) {
  return MlirOpBuilder {.ptr = new OpBuilderT(unwrap(ctx))};
}

MlirOpBuilder mlirOpBuilderCreateWithListener(MlirContext ctx, MlirOpBuilderListener listener) {
  auto *l = reinterpret_cast<ListenerT *>(listener.ptr);
  return MlirOpBuilder {.ptr = new OpBuilderT(unwrap(ctx), l)};
}

void mlirOpBuilderDestroy(MlirOpBuilder builder) {
  delete reinterpret_cast<OpBuilderT *>(builder.ptr);
}

MlirContext mlirOpBuilderGetContext(MlirOpBuilder builder) {
  return wrap(unwrap(builder)->getContext());
}

void mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder, MlirBlock block) {
  unwrap(builder)->setInsertionPointToStart(unwrap(block));
}

MlirOperation mlirOpBuilderGetInsertionPoint(MlirOpBuilder builder) {
  auto it = unwrap(builder)->getInsertionPoint();
  auto *blk = unwrap(builder)->getInsertionBlock();
  if (!blk) {
    return MlirOperation {nullptr};
  }

  return wrap(it != blk->end() ? &*it : nullptr);
}

/// Returns the current insertion block in the builder.
MlirBlock mlirOpBuilderGetInsertionBlock(MlirOpBuilder builder) {
  return wrap(unwrap(builder)->getInsertionBlock());
}

MlirOperation mlirOpBuilderInsert(MlirOpBuilder builder, MlirOperation op) {
  return wrap(unwrap(builder)->insert(unwrap(op)));
}

//===----------------------------------------------------------------------===//
// MlirOpBuilderListener
//===----------------------------------------------------------------------===//

MlirOpBuilderListener mlirOpBuilderListenerCreate(
    MlirNotifyOperationInserted opCb, MlirNotifyBlockInserted blockCb, void *userData
) {
  return MlirOpBuilderListener {.ptr = new ListenerT(opCb, blockCb, userData)};
}

void mlirOpBuilderListenerDestroy(MlirOpBuilderListener listener) {
  delete reinterpret_cast<ListenerT *>(listener.ptr);
}
