//===-- OpTraits.td ----------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FUNCTION_TRAITS
#define LLZK_FUNCTION_TRAITS

include "mlir/IR/OpBase.td"

/// Marker for ops that are specific to witness generation. Op classes with
/// this trait are valid only in functions that have the `AllowWitnessAttr`.
def WitnessGen : NativeOpTrait<"WitnessGen">, StructuralOpTrait {
  string cppNamespace = "::llzk::function";
}

/// Marker for ops that are specific to constraint generation. Op classes with
/// this trait are valid only in functions that have the `AllowConstraintAttr`.
def ConstraintGen : NativeOpTrait<"ConstraintGen">, StructuralOpTrait {
  string cppNamespace = "::llzk::function";
}

/// Marker for ops over `llzk.felt` type operands that are not native to finite
/// field arithmetic and thus must be rewritten or folded into a constant after
/// the flattening and unrolling pass. Op classes with this trait are valid only
/// in functions that have the `AllowNonNativeFieldOpsAttr`.
def NotFieldNative : NativeOpTrait<"NotFieldNative">, StructuralOpTrait {
  string cppNamespace = "::llzk::function";
}

#endif // LLZK_FUNCTION_TRAITS
