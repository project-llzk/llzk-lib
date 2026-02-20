//===-- Builder.h - C API for op builder --------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares a type that supports the creation of operations and
// handles their insertion in blocks.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_BUILDER_H
#define LLZK_C_BUILDER_H

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEFINE_C_API_STRUCT(name, storage)                                                         \
  struct name {                                                                                    \
    storage *ptr;                                                                                  \
  };                                                                                               \
  typedef struct name name

DEFINE_C_API_STRUCT(MlirOpBuilder, void);
DEFINE_C_API_STRUCT(MlirOpBuilderListener, void);

#undef DEFINE_C_API_STRUCT

struct MlirOpBuilderInsertPoint {
  MlirBlock block;
  MlirOperation point;
};
typedef struct MlirOpBuilderInsertPoint MlirOpBuilderInsertPoint;

typedef void (*MlirNotifyOperationInserted)(MlirOperation, MlirOpBuilderInsertPoint, void *);
typedef void (*MlirNotifyBlockInserted)(MlirBlock, MlirRegion, MlirBlock, void *);

//===----------------------------------------------------------------------===//
// MlirOpBuilder
//===----------------------------------------------------------------------===//

// The API for OpBuilder is left barebones for now since we only need a reference that we can pass
// to op build methods that we expose. More methods can be added as the need for them arises.

#define DECLARE_SUFFIX_OP_BUILDER_CREATE_FN(suffix, ...)                                           \
  MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderCreate##suffix(__VA_ARGS__);                       \
  MLIR_CAPI_EXPORTED MlirOpBuilder mlirOpBuilderCreate##suffix##WithListener(                      \
      __VA_ARGS__, MlirOpBuilderListener                                                           \
  );
#define DECLARE_OP_BUILDER_CREATE_FN(...) DECLARE_SUFFIX_OP_BUILDER_CREATE_FN(, __VA_ARGS__)

DECLARE_OP_BUILDER_CREATE_FN(MlirContext context)

#undef DECLARE_OP_BUILDER_CREATE_FN

/// Destroys the given builder.
MLIR_CAPI_EXPORTED void mlirOpBuilderDestroy(MlirOpBuilder builder);

/// Returns the context.
MLIR_CAPI_EXPORTED MlirContext mlirOpBuilderGetContext(MlirOpBuilder builder);

/// Sets the insertion point to the beginning of the given block.
MLIR_CAPI_EXPORTED void
mlirOpBuilderSetInsertionPointToStart(MlirOpBuilder builder, MlirBlock block);

/// Returns the current insertion point in the builder.
MLIR_CAPI_EXPORTED MlirOperation mlirOpBuilderGetInsertionPoint(MlirOpBuilder builder);

/// Returns the current insertion block in the builder.
MLIR_CAPI_EXPORTED MlirBlock mlirOpBuilderGetInsertionBlock(MlirOpBuilder builder);

/// Insert the given operation at the current insertion point and return it.
MLIR_CAPI_EXPORTED MlirOperation mlirOpBuilderInsert(MlirOpBuilder builder, MlirOperation op);

//===----------------------------------------------------------------------===//
// MlirOpBuilderListener
//===----------------------------------------------------------------------===//

/// Creates a new mlir::OpBuilder::Listener. Takes one callback for each method of the Listener
/// interface and a pointer to user defined data.
MLIR_CAPI_EXPORTED MlirOpBuilderListener mlirOpBuilderListenerCreate(
    MlirNotifyOperationInserted operationCb, MlirNotifyBlockInserted blockCb, void *userData
);

/// Destroys the given listener.
MLIR_CAPI_EXPORTED void mlirOpBuilderListenerDestroy(MlirOpBuilderListener listener);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_BUILDER_H
