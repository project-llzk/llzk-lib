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

def StructCleanupModeDescription {
  string r = "Specifies the extent to which unused parameterized definitions (i.e. "
             "structs or free functions within a `poly.template`) are removed during the "
             "flattening pass.";
}

def StructCleanupMode
    : I32EnumAttr<"StructCleanupMode", StructCleanupModeDescription.r,
                  [
                      // Disabled: No definitions are deleted.
                      I32EnumAttrCase<"Disabled", 0, "disabled">,
                      // Preimage: Only definitions that were replaced with concrete
                      // instantiations are deleted.
                      I32EnumAttrCase<"Preimage", 1, "preimage">,
                      // ConcreteAsRoot: All definitions that cannot be reached by a
                      // use-def chain from some concrete definition are deleted.
                      I32EnumAttrCase<"ConcreteAsRoot", 2, "concrete-as-root">,
                      // MainAsRoot: All definitions that cannot be reached by a
                      // use-def chain from the main struct are deleted.
                      I32EnumAttrCase<"MainAsRoot", 3, "main-as-root">,
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
  let constructor = "llzk::polymorphic::createEmptyTemplateRemoval()";
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
  let constructor = "llzk::polymorphic::createFlatteningPass()";

  let options = [Option<
                     "iterationLimit", "max-iter", "unsigned",
                     /* default */ "1000",
                     "Maximum number of times the pass will run if a fixpoint "
                     "is not reached earlier. Unrolling loops can provide more "
                     "opportunities for instantiating structs but the converse "
                     "is true as well. Thus, the pass will run multiple times "
                     "until no further changes can be made or the upper limit "
                     "provided in this option is reached.">,
                 Option<"cleanupMode", "cleanup",
                        "::llzk::polymorphic::StructCleanupMode",
                        /* default */
                        "::llzk::polymorphic::StructCleanupMode::Preimage",
                        StructCleanupModeDescription.r, [{::llvm::cl::values(
                clEnumValN(::llzk::polymorphic::StructCleanupMode::Disabled,
                           stringifyStructCleanupMode(::llzk::polymorphic::StructCleanupMode::Disabled), "No definitions are deleted."),
                clEnumValN(::llzk::polymorphic::StructCleanupMode::Preimage,
                           stringifyStructCleanupMode(::llzk::polymorphic::StructCleanupMode::Preimage),
                           "Only definitions that were replaced with concrete instantiations are deleted."),
                clEnumValN(::llzk::polymorphic::StructCleanupMode::ConcreteAsRoot,
                           stringifyStructCleanupMode(::llzk::polymorphic::StructCleanupMode::ConcreteAsRoot),
                           "All definitions that cannot be reached by a use-def chain from some concrete definition are deleted."),
                clEnumValN(::llzk::polymorphic::StructCleanupMode::MainAsRoot,
                           stringifyStructCleanupMode(::llzk::polymorphic::StructCleanupMode::MainAsRoot),
                           "All definitions that cannot be reached by a use-def chain from the \"Main\" struct are deleted.")
          )}]>,
  ];
}

#endif // LLZK_POLYMORPHIC_TRANSFORMATION_PASSES_TD
