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

/// Marker for ops in the `verif` dialect that can be inlined inside functions.
/// The op can be marked with additional traits that must be satisfied if the op 
/// is inlined. The difference between this and passing the attribute as normal 
/// is that the extra traits are only checked if the op is inlined. This allows 
/// marking the op with very restrictive traits like `ConstraintGen`. If that trait 
/// was naively added to the inlinable op it will fail verification when the op 
/// is not inlined.
///
/// For example, an op that can only be inlined inside constrain functions can
/// be marked with `Verification<["::llzk::function::ConstraintGen"]>`.
class Verification<list<string> extra = []>
    : ParamNativeOpTrait<"Verification", !interleave(extra, ",")>,
      StructuralOpTrait {
  string cppNamespace = "::llzk::function";
}



#endif // LLZK_FUNCTION_TRAITS
