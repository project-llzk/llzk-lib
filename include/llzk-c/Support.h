//===-- Support.h - C API general utilities -----------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares utilities for working with the C API.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_IR_H
#define LLZK_C_IR_H

#include "llzk-c/Builder.h" // IWYU pragma: keep

#include <mlir-c/IR.h> // IWYU pragma: keep
#include <mlir-c/Support.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Utility macros for function declarations.
//===----------------------------------------------------------------------===//

#define LLZK_BUILD_METHOD_NAME(dialect, op, suffix) llzk##dialect##_##op##Build##suffix
#define LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(dialect, op, suffix, ...)                              \
  MLIR_CAPI_EXPORTED MlirOperation LLZK_BUILD_METHOD_NAME(dialect, op, suffix)(                    \
      MlirOpBuilder builder, MlirLocation location, __VA_ARGS__                                    \
  )
// Used for when the build method is "general" and does not have a suffix at the end.
#define LLZK_DECLARE_OP_BUILD_METHOD(dialect, op, ...)                                             \
  LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(dialect, op, , __VA_ARGS__)

#define LLZK_DECLARE_PREDICATE(name, ...) MLIR_CAPI_EXPORTED bool llzk##name(__VA_ARGS__)

#define LLZK_DECLARE_OP_PREDICATE(op, name)                                                        \
  MLIR_CAPI_EXPORTED bool llzk##op##Get##name(MlirOperation op)
#define LLZK_DECLARE_NARY_OP_PREDICATE(op, name, ...)                                              \
  MLIR_CAPI_EXPORTED bool llzk##op##Get##name(MlirOperation op, __VA_ARGS__)

#define LLZK_DECLARE_ISA(dialect, what, root)                                                      \
  MLIR_CAPI_EXPORTED bool llzk##root##IsA##_##dialect##_##what(Mlir##root what)
#define LLZK_DECLARE_TYPE_ISA(dialect, what) LLZK_DECLARE_ISA(dialect, what, Type)
#define LLZK_DECLARE_OP_ISA(dialect, what) LLZK_DECLARE_ISA(dialect, what, Operation)
#define LLZK_DECLARE_ATTR_ISA(dialect, what) LLZK_DECLARE_ISA(dialect, what, Attribute)

//===----------------------------------------------------------------------===//
// Representation of a mlir::ValueRange.
//===----------------------------------------------------------------------===//

struct MlirValueRange {
  MlirValue const *values;
  intptr_t size;
};
typedef struct MlirValueRange MlirValueRange;

//===----------------------------------------------------------------------===//
// Symbol lookup result.
//===----------------------------------------------------------------------===//

typedef struct LlzkSymbolLookupResult {
  void *ptr;
} LlzkSymbolLookupResult;

/// Destroys the lookup result, releasing its resources.
MLIR_CAPI_EXPORTED void llzkSymbolLookupResultDestroy(LlzkSymbolLookupResult result);

/// Returns the looked up Operation.
///
/// The lifetime of the Operation is tied to the lifetime of the lookup result.
MLIR_CAPI_EXPORTED MlirOperation LlzkSymbolLookupResultGetOperation(LlzkSymbolLookupResult result);

//===----------------------------------------------------------------------===//
// MLIR ports.
//===----------------------------------------------------------------------===//

/// Replace uses of 'of' value with the 'with' value inside the 'op' operation.
/// Note: Duplicated from upstream LLVM. Available in 21.1.8 and later.
MLIR_CAPI_EXPORTED void
mlirOperationReplaceUsesOfWith(MlirOperation op, MlirValue of, MlirValue with);

//===----------------------------------------------------------------------===//
// CAPI support of additional MLIR functionality.
//===----------------------------------------------------------------------===//

/// Walks operation `op` in `walkOrder`, with operations at the same nesting level traversed in
/// reverse order, and calls `callback` on that operation. `*userData` is passed to the callback as
/// well and can be used to tunnel some context or other data into the callback.
MLIR_CAPI_EXPORTED
void mlirOperationWalkReverse(
    MlirOperation from, MlirOperationWalkCallback callback, void *userData, MlirWalkOrder walkOrder
);

//===----------------------------------------------------------------------===//
// Helper types and functions for map operands constructor arguments.
//===----------------------------------------------------------------------===//

/// Encapsulates the arguments related to affine maps that are common in operation constructors that
/// support them.
typedef struct LlzkAffineMapOperandsBuilder {
  intptr_t nMapOperands;
  // A list of lists of Values. The outer list is owned by this struct but the inner lists
  // are not and are considered views (similar to the C++ `ValueRange` class).
  MlirValueRange *mapOperands;
  /// Set to a negative number to indicate that `dimsPerMap.attr` must be used
  /// instead of `dimsPerMap.array`.
  /// The default mode is `Array` with an uninitialized array.
  intptr_t nDimsPerMap;
  union {
    MlirAttribute attr;
    /// List of dimension counts. Is owned by this struct and it gets automatically
    /// allocated/deallocated if the user switches to or from `Attr` mode.
    int32_t *array;
  } dimsPerMap;
} LlzkAffineMapOperandsBuilder;

/// Creates a new struct. The owner is responsible for cleaning the struct.
MLIR_CAPI_EXPORTED LlzkAffineMapOperandsBuilder llzkAffineMapOperandsBuilderCreate(void);

/// Destroys the struct releasing its resources.
MLIR_CAPI_EXPORTED void llzkAffineMapOperandsBuilderDestroy(LlzkAffineMapOperandsBuilder *builder);

/// Appends the value ranges to the list of map operands.
MLIR_CAPI_EXPORTED void llzkAffineMapOperandsBuilderAppendOperands(
    LlzkAffineMapOperandsBuilder *builder, intptr_t n, MlirValueRange const *mapOperands
);

/// Appends the value ranges to the list of map operands and indicates how many of these operands
/// are dimensions.
///
/// Asserts that the number of map operands and the number of dimensions per map is the same to
/// avoid going out of sync.
MLIR_CAPI_EXPORTED void llzkAffineMapOperandsBuilderAppendOperandsWithDimCount(
    LlzkAffineMapOperandsBuilder *builder, intptr_t n, MlirValueRange const *mapOperands,
    int32_t const *dimsPerMap
);

/// Appends a dimension count to the list of dimensions per map.
///
/// If the builder is in `Attr` mode allocates an array with enough space and copies the attr's
/// contents into it before appending the new value.
MLIR_CAPI_EXPORTED void llzkAffineMapOperandsBuilderAppendDimCount(
    LlzkAffineMapOperandsBuilder *builder, intptr_t n, int32_t const *dimsPerMap
);

/// Sets the number of dimensions per map to the given attribute. Overwrites the previous values and
/// deallocates if necessary.
///
/// Asserts that the attribute is of type `DenseI32ArrayAttr`.
MLIR_CAPI_EXPORTED void llzkAffineMapOperandsBuilderSetDimsPerMapFromAttr(
    LlzkAffineMapOperandsBuilder *builder, MlirAttribute attribute
);

/// Converts the list of dimensions defined as an attribute into an array.
///
/// This function is a no-op if the list was already in array mode.
MLIR_CAPI_EXPORTED void
llzkAffineMapOperandsBuilderConvertDimsPerMapToArray(LlzkAffineMapOperandsBuilder *builder);

/// Converts the list of dimensions defined as an array into an attribute.
///
/// This function is a no-op if the list was already in attribute mode.
MLIR_CAPI_EXPORTED void llzkAffineMapOperandsBuilderConvertDimsPerMapToAttr(
    LlzkAffineMapOperandsBuilder *builder, MlirContext context
);

/// Returns the number of dimensions per map represented as an attribute.
MLIR_CAPI_EXPORTED MlirAttribute llzkAffineMapOperandsBuilderGetDimsPerMapAttr(
    LlzkAffineMapOperandsBuilder builder, MlirContext context
);

#ifdef __cplusplus
}
#endif

#endif
