//===-- TransformationPasses.td ------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POLYMORPHIC_TRANSFORMATION_PASSES_TD
#define LLZK_POLYMORPHIC_TRANSFORMATION_PASSES_TD

include "llzk/Pass/PassBase.td"
include "mlir/IR/EnumAttr.td"

def FlatteningCleanupModeDescription {
  string r =
      "Specifies the extent to which unused parameterized definitions (i.e. "
      "structs or free functions within a `poly.template`) are removed during "
      "the flattening pass.";
}

def FlatteningCleanupMode
    : I32EnumAttr<"FlatteningCleanupMode", FlatteningCleanupModeDescription.r,
                  [
                      // Unspecified: Leave the cleanup choice to the caller
                      // (defaults to `preimage` if not specified).
                      I32EnumAttrCase<"Unspecified", 0, "unspecified">,
                      // Disabled: No definitions are deleted.
                      I32EnumAttrCase<"Disabled", 1, "disabled">,
                      // Preimage: Only definitions that were replaced with
                      // concrete instantiations are deleted.
                      I32EnumAttrCase<"Preimage", 2, "preimage">,
                      // ConcreteAsRoot: All definitions that cannot be reached
                      // by a use-def chain from some concrete definition are
                      // deleted.
                      I32EnumAttrCase<"ConcreteAsRoot", 3, "concrete-as-root">,
                      // MainAsRoot: All definitions that cannot be reached by a
                      // use-def chain from the main struct are deleted.
                      I32EnumAttrCase<"MainAsRoot", 4, "main-as-root">,
]> {
  let cppNamespace = "::llzk::polymorphic";
  let genSpecializedAttr = 0;
}

def EmptyTemplateRemovalPass : LLZKPass<"llzk-drop-empty-templates"> {
  let summary = "Remove empty templates";
  let description = [{
    Performs the following transformations:
    - Convert templates with no constant parameters or expressions into modules.
    - Remove templates with no struct or function definitions.
  }];
}

def FlatteningPass : LLZKPass<"llzk-flatten"> {
  let summary =
      "Instantiate templates, unroll loops, and propagate instantiated types";
  let description = [{
    Iterates the following transformations to a fixpoint, bounded by `max-iter`:

    - Specialize referenced parameterized structs and free functions from concrete template
      arguments, signature unification, or wildcard type parameters inferred from the body.
      When only some bindings are concrete, the result is a reduced `poly.template` containing the
      remaining non-concrete parameter bindings.
    - Evaluate referenced `poly.expr` bindings once their value dependencies and operation
      operand/result type dependencies are concrete. A referenced expression with remaining
      symbolic dependencies is retained in that reduced template after known value and type
      bindings are substituted. If a known binding's substituted type remains non-concrete, that
      specialization attempt is deferred. Once every dependency is concrete, an operation that is
      unsafe to speculate, has memory effects, or fails to fold is an error. A definition directly
      inside a template with `poly.expr` bindings and no `poly.param` operations is specialized
      when it references a concrete expression.
    - Propagate concrete bindings through type variables, array dimensions and element
      types, parameterized struct arguments, POD record types, and nested-call arguments.
    - Unroll loops with known bounds, instantiate affine-map parameters, and propagate resulting
      types.

    A specialization is kept only after body conversion succeeds; free-function clones must also
    pass nested-call verification. After the fixpoint, the selected cleanup mode is applied.
  }];
  // Implementation note: These options should be kept in sync with
  // `StructInliningFlatteningOptions` in `LLZKTransformationPassPipelines.h`.
  let options =
      [Option<"iterationLimit", "max-iter", "unsigned",
              /* default */ "1000",
              "Maximum number of times the pass will run if a fixpoint "
              "is not reached earlier. Unrolling loops can provide more "
              "opportunities for instantiating structs but the converse "
              "is true as well. Thus, the pass will run multiple times "
              "until no further changes can be made or the upper limit "
              "provided in this option is reached.">,
       Option<"cleanupMode", "cleanup",
              "::llzk::polymorphic::FlatteningCleanupMode",
              /* default */
              "::llzk::polymorphic::FlatteningCleanupMode::Unspecified",
              FlatteningCleanupModeDescription.r, [{::llvm::cl::values(
                clEnumValN(::llzk::polymorphic::FlatteningCleanupMode::Unspecified,
                           stringifyFlatteningCleanupMode(::llzk::polymorphic::FlatteningCleanupMode::Unspecified),
                           "Use the cleanup mode specified by the calling pipeline (defaults to `preimage` if not specified)."),
                clEnumValN(::llzk::polymorphic::FlatteningCleanupMode::Disabled,
                           stringifyFlatteningCleanupMode(::llzk::polymorphic::FlatteningCleanupMode::Disabled),
                           "No definitions are deleted."),
                clEnumValN(::llzk::polymorphic::FlatteningCleanupMode::Preimage,
                           stringifyFlatteningCleanupMode(::llzk::polymorphic::FlatteningCleanupMode::Preimage),
                           "Only definitions that were replaced with concrete instantiations are deleted."),
                clEnumValN(::llzk::polymorphic::FlatteningCleanupMode::ConcreteAsRoot,
                           stringifyFlatteningCleanupMode(::llzk::polymorphic::FlatteningCleanupMode::ConcreteAsRoot),
                           "All definitions that cannot be reached by a use-def chain from some concrete definition are deleted."),
                clEnumValN(::llzk::polymorphic::FlatteningCleanupMode::MainAsRoot,
                           stringifyFlatteningCleanupMode(::llzk::polymorphic::FlatteningCleanupMode::MainAsRoot),
                           "All definitions that cannot be reached by a use-def chain from the \"Main\" struct are deleted.")
          )}]>,
  ];
}

def WildcardArraySpecializationPass
    : LLZKPass<"llzk-specialize-wildcard-arrays"> {
  let summary =
      "Refine wildcard array casts and specialize concrete call targets";
  let description = [{
    Refines `poly.unifiable_cast` results when wildcard `array.type` dimensions can be replaced
    with concrete integer sizes from the input type. Then specializes free functions, extern
    declarations, and whole structs for calls whose wildcard array dimensions have become concrete.

    This pass is intended to run after `llzk-flatten`. It iterates to a fixpoint so newly-refined
    cast result types can enable further callable specialization in later iterations.
  }];
  let options = [Option<"iterationLimit", "max-iter", "unsigned",
                        /* default */ "1000",
                        "Maximum number of iterations before the pass gives up "
                        "reaching a fixpoint.">];
}

def TypeVarInferencePass : LLZKPass<"llzk-infer-tvar"> {
  let summary = "Infer concrete function types for polymorphic type variables";
  let description = [{
    Infers concrete replacements for `!poly.tvar` template parameters from operations in
    `function.def` or `poly.expr` bodies, such as `poly.unifiable_cast` between a type variable
    and a more concrete type (which means for maximum effect you should run it before other
    optimization passes that can remove `poly.unifiable_cast`). Rewrites affected function
    signatures and body types, removes redundant casts, drops resolved `poly.param` declarations,
    and updates call-site template parameter lists.

    The pass also creates template-local instantiated versions of functions for each
    type-variable replacement. The `llzk-flatten` pass performs similar instantiation of type
    variables but only as part of its more extensive lowering and flattening, all of which may
    not be desired in some cases (hence the utility of this pass).

    This pass may result in `poly.template` containing no `poly.param` or `poly.expr`. Run
    `llzk-drop-empty-templates` after this pass to simplify these templates.
  }];
}

#endif // LLZK_POLYMORPHIC_TRANSFORMATION_PASSES_TD
