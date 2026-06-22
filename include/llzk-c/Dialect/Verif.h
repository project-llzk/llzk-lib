//===-- Verif.h - C API for Verif dialect -------------------------*- C -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface for registering and accessing the
// Verif dialect. A dialect should be registered with a context to make it
// available to users of the context. These users must load the dialect
// before using any of its attributes, operations, or types. Parser and pass
// manager can load registered dialects automatically.
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_C_DIALECT_VERIF_H
#define LLZK_C_DIALECT_VERIF_H

#include "llzk-c/Support.h"

#include <mlir-c/IR.h>
#include <mlir-c/Support.h>

// Include the generated CAPI
#include "llzk/Dialect/Verif/IR/Ops.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

/// Get reference to the LLZK `verif` dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Verif, llzk__verif);

/// Attaches the interfaces defined by the `verif` dialect to upstream IR elements
///
/// Attempting to use those interfaces without calling this function first will result in an error.
MLIR_CAPI_EXPORTED void llzkVerif_attachInterfaces(MlirContext context);

/// Build a `verif.contract` from explicit attributes and signature metadata.
LLZK_DECLARE_OP_BUILD_METHOD(
    Verif, ContractOp, MlirIdentifier sym_name, MlirAttribute target, MlirAttribute function_type,
    MlirAttribute arg_attrs
);

/// Build a `verif.contract` with arguments and attributes derived from the target.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Verif, ContractOp, FromTargetIdentifier, MlirIdentifier sym_name, MlirIdentifier target
);

/// Build a `verif.contract` with arguments and attributes derived from the target.
LLZK_DECLARE_SUFFIX_OP_BUILD_METHOD(
    Verif, ContractOp, FromTargetAttr, MlirIdentifier sym_name, MlirAttribute target
);

/// Build a `verif.include` with a flat list of call operands and no affine-map
/// instantiations.
LLZK_DECLARE_OP_BUILD_METHOD(
    Verif, IncludeOp, MlirAttribute callee, MlirValueRange argOperands, MlirAttribute templateParams
);

/// Build a `verif.invariant` with a list of argument types and locations.
///
/// The pointers to the types and locations must point to the same amount of elements.
LLZK_DECLARE_OP_BUILD_METHOD(
    Verif, InvariantOp, MlirStringRef loopName, intptr_t numArgs, MlirType const *types,
    MlirLocation const *locs
);

/// Returns the body of the invariant operation.
MLIR_CAPI_EXPORTED MlirBlock llzkVerif_InvariantOpGetBody(MlirOperation op);

#ifdef __cplusplus
}
#endif

#endif // LLZK_C_DIALECT_VERIF_H
