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

/// Concrete implementation of `mlir::OpBuilder::Listener` that delegates
/// notifications to user-supplied C callbacks.
class ListenerT : public OpBuilder::Listener {
public:
  ListenerT(MlirNotifyOperationInserted op, MlirNotifyBlockInserted block, void *data)
      : opInsertedCb(op), blockInsertedCb(block), userData(data) {}

  /// Called after an operation is inserted. Wraps `op` and the previous insert
  /// point and forwards them to the `MlirNotifyOperationInserted` callback.
  void notifyOperationInserted(Operation *op, OpBuilder::InsertPoint previous) final {
    MlirOpBuilderInsertPoint i {
        .block = wrap(previous.getBlock()), .point = wrap(&*previous.getPoint())
    };
    opInsertedCb(wrap(op), i, userData);
  }

  /// Called after a block is inserted. Wraps `block`, `previous` region, and the
  /// iterator position, then forwards them to the `MlirNotifyBlockInserted` callback.
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

/// Creates a new `OpBuilder` for the given MLIR context.
MlirOpBuilder mlirOpBuilderCreate(MlirContext ctx) {
  return MlirOpBuilder {.ptr = new OpBuilderT(unwrap(ctx))};
}

/// Creates a new `OpBuilder` for the given MLIR context and attaches `listener`
/// so that operation and block insertion events are forwarded to its callbacks.
MlirOpBuilder mlirOpBuilderCreateWithListener(MlirContext ctx, MlirOpBuilderListener listener) {
  auto *l = reinterpret_cast<ListenerT *>(listener.ptr);
  return MlirOpBuilder {.ptr = new OpBuilderT(unwrap(ctx), l)};
}

/// Destroys `builder` and frees the underlying `OpBuilder` object.
void mlirOpBuilderDestroy(MlirOpBuilder builder) {
  delete reinterpret_cast<OpBuilderT *>(builder.ptr);
}

/// Returns the MLIR context associated with `builder`.
MlirContext mlirOpBuilderGetContext(MlirOpBuilder builder) {
  return wrap(unwrap(builder)->getContext());
}

/// Sets the insertion point of `builder` to the beginning of `block` so
/// that subsequent insertions prepend to that block.
void mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder, MlirBlock block) {
  unwrap(builder)->setInsertionPointToStart(unwrap(block));
}

/// Returns the operation before which new operations will be inserted, or a null `MlirOperation` if
/// there is no current insertion block or the insertion point is at the end of the block.
MlirOperation mlirOpBuilderGetInsertionPoint(MlirOpBuilder builder) {
  auto it = unwrap(builder)->getInsertionPoint();
  auto *blk = unwrap(builder)->getInsertionBlock();
  if (!blk) {
    return MlirOperation {nullptr};
  }

  return wrap(it != blk->end() ? &*it : nullptr);
}

/// Returns the block that `builder` is currently inserting into, or a null
/// `MlirBlock` if no insertion point has been set.
MlirBlock mlirOpBuilderGetInsertionBlock(MlirOpBuilder builder) {
  return wrap(unwrap(builder)->getInsertionBlock());
}

/// Inserts `op` at the current insertion point of `builder` and returns it.
MlirOperation mlirOpBuilderInsert(MlirOpBuilder builder, MlirOperation op) {
  return wrap(unwrap(builder)->insert(unwrap(op)));
}

//===----------------------------------------------------------------------===//
// MlirOpBuilderListener
//===----------------------------------------------------------------------===//

/// Creates a new `OpBuilder::Listener` that calls `opCb` whenever an operation
/// is inserted and `blockCb` whenever a block is inserted. `userData` is an
/// opaque pointer forwarded unchanged to each callback.
MlirOpBuilderListener mlirOpBuilderListenerCreate(
    MlirNotifyOperationInserted opCb, MlirNotifyBlockInserted blockCb, void *userData
) {
  return MlirOpBuilderListener {.ptr = new ListenerT(opCb, blockCb, userData)};
}

/// Destroys `listener` and frees the underlying `ListenerT` object.
void mlirOpBuilderListenerDestroy(MlirOpBuilderListener listener) {
  delete reinterpret_cast<ListenerT *>(listener.ptr);
}
