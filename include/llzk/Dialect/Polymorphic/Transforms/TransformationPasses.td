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

def TemplateMonomorphizationPass : LLZKPass<"llzk-monomorphize-templates"> {
  let summary = "Specialize structs and free functions reachable from llzk.main";
  let description = [{
    Starting with a concrete `llzk.main`, discover and instantiate struct and
    free-function dependencies until every requested specialization has been
    processed. Reuse definitions by source symbol and concrete argument list,
    including reachable definitions with no template parameters. Free-function
    arguments may be explicit on a call or inferred from its type signature.
    Preserve source templates and control flow, and update direct type uses and
    calls to their concrete definitions. Free-function calls retained in
    unselected branches are also specialized so their rewritten types match
    the callee signature.

    Loops remain in the IR rather than being unrolled; discovery evaluates their
    iterations to identify specializations without expanding the loop bodies.
    "Rolled" calls and struct types below retain affine parameters that depend on
    those iterations or on array element indices.

    Affine calls and array values retain their parameterized types. Their
    `poly.family_specializations` and `poly.family` attributes identify the
    concrete definitions needed by later elaboration; definitions carry
    `poly.specialization_id`, `poly.origin` and `poly.arguments` attributes.
    Free-function clones carry `poly.origin` and `poly.arguments`; their
    identities are separate from the struct IDs used by families.
    `poly.family` appears on members and operations producing an array value,
    including `struct.readm` and `array.new`. Each entry names the rolled struct
    type, the enclosing array's index tuple, and the matching specialization.
    The `type` entry retains the original affine-parameterized family type; it
    identifies which struct field is being specialized within a POD element.
    `specialization` identifies the concrete clone through its
    `poly.specialization_id`. For example, an element at index 2 of a Child<i>
    family retains Child<i> in `type` and names the clone whose `poly.arguments`
    are [2]. Concrete sibling fields have no family entry.
    On a rolled call, `poly.family_specializations` lists candidates collected
    from all matching arrays and affine calls in the source instance. The list
    may be empty only when a known array has an empty dimension. A consumer
    selects a target by evaluating the argument type's affine map using the
    value's affine inputs, often an array element's indices, and matching the
    resulting arguments to `poly.arguments`.
    `poly.origin` and every struct name nested in `poly.arguments` are relative
    to the outermost `llzk.lang` root, which is also the lookup anchor when
    reloading this metadata. Cache keys use the same argument spelling, so
    requests from different roots reuse the same specialization. Substitution
    re-spells type arguments relative to the clone's root; an argument whose
    definition is outside that root is diagnosed at the instantiation site.
    Arguments to typed `poly.param` declarations are converted to the declared
    type using the conversions below, before forming a cache key. An
    incompatible argument is diagnosed. Felt-to-index conversion requires a
    value in the nonnegative signed 64-bit index range. Arguments to felt
    parameters with a known field are reduced modulo its prime before forming
    the cache key and substituting uses. A felt literal with no specified field
    acquires the parameter's declared field; a different explicit field is
    rejected. Equivalent field values therefore share a specialization. With no
    known field, index-to-felt conversion requires a nonnegative value and
    preserves it. Untyped parameters retain their input
    attributes, so differently spelled numeric values can create distinct
    specializations. An `i1` parameter accepts only boolean arguments or index
    zero and one, which become false and true. Other index values fail type
    conversion at the instantiation site. The normalized boolean binding
    determines specialization identity and is used for every read and forwarded
    argument.
    Forwarding that value to an index parameter produces zero or one.
    A felt binding must be read with its exact felt type, including its field,
    even for untyped parameters. Mismatched felt types are diagnosed at
    `poly.read_const`.
    Reading it as index or i1 is diagnosed at `poly.read_const`; use an explicit
    cast after a felt read for numeric conversion. Integer bindings read as i1
    must be zero or one. Integer-to-felt reads use the same modular conversion
    as typed arguments, and negative values require a known field modulus.
    Reads whose type is a type variable use its concrete binding.

    Discovered methods and free functions must be defined and single-block.
    Supported regions are `scf.if`,
    signed `scf.for` with known bounds and positive steps, and `scf.while` with
    a known condition at every check. Unstructured branches are diagnosed.
    Constant evaluation
    uses template bindings and fold hooks; global and aggregate reads and call
    results remain unknown. Required specialization arguments must resolve.
    External free-function templates are unsupported.
    Includes must be inlined before running this pass.
  }];
  let options =
      [Option<"specializationLimit", "max-specializations", "unsigned", "1000",
              "Maximum number of concrete struct and free-function definitions">,
       Option<"evaluationLimit", "max-evaluation-steps", "uint64_t", "1000000",
              "Maximum steps shared by template expressions and methods in one "
              "discovery run">];
}

def FlatteningPass : LLZKPass<"llzk-flatten"> {
  let summary = "Flatten structs and unroll loops";
  let description = [{
    Performs the following transformations:
    - Instantiate `affine_map` parameters of StructType and ArrayType
      to constant values using the arguments at the instantiation site
    - Replace parameterized structs with flattened (i.e., no parameter)
      versions of those structs based on requested return type at calls
      to `compute()` functions and unroll loops
    - Unroll loops
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
