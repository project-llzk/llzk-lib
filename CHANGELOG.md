## v2.0.0 - 2026-04-10
### Added
- 'Interval analysis support for `arith.select`, `arith.xori`, `felt.uintdiv`, `felt.sintdiv`, and `felt.bit_or`'
- 'Interval analysis support for `array.new`, including empty-array roots and explicit element initializers'
- 'Interval analysis support for `array.write` so later `array.read` operations can reuse written intervals'
- 'Support for `bool` dialect operators to `pcl` boolean ops'
- '`--llzk-enforce-no-overwrite` pass to detect possible struct member overwrites and possible uninstantiated struct members'
- '`CallOp::unifyTypeSignature()` method'
- '`podTypesUnify()` and `functionTypesUnify()` functions in `TypeHelper`'
- '`poly.template`, `poly.param`, and `poly.expr` ops'
- 'Folding implementation for `bool` and `felt` operations'
- Functions to query symbol uses within types
- Automatic IR migration from version 1.x.x to 2.0.0
- CAPI:
  - '`llzkStruct_StructDefOpGetTemplateParamOpNames`'
  - '`llzkStruct_StructDefOpGetNumTemplateParamOpNames`'
  - '`llzkStruct_StructDefOpGetTemplateExprOpNames`'
  - '`llzkStruct_StructDefOpGetNumTemplateExprOpNames`'
  - '`llzkOperationIsA_Poly_TemplateOp`'
  - '`llzkPoly_TemplateOpBuild`'
  - '`llzkPoly_TemplateOpGetSymName`'
  - '`llzkPoly_TemplateOpSetSymName`'
  - '`llzkPoly_TemplateOpGetBody`'
  - '`llzkPoly_TemplateOpGetBodyRegion`'
  - '`llzkPoly_TemplateOpHasConstParamOps`'
  - '`llzkPoly_TemplateOpNumConstParamOps`'
  - '`llzkPoly_TemplateOpGetConstParamNames`'
  - '`llzkPoly_TemplateOpHasConstParamNamed`'
  - '`llzkPoly_TemplateOpHasConstExprOps`'
  - '`llzkPoly_TemplateOpNumConstExprOps`'
  - '`llzkPoly_TemplateOpGetConstExprNames`'
  - '`llzkPoly_TemplateOpHasConstExprNamed`'
  - '`llzkOperationIsA_Poly_TemplateExprOp`'
  - '`llzkPoly_TemplateExprOpBuild`'
  - '`llzkPoly_TemplateExprOpGetSymName`'
  - '`llzkPoly_TemplateExprOpSetSymName`'
  - '`llzkPoly_TemplateExprOpGetInitializerRegion`'
  - '`llzkPoly_TemplateExprOpGetType`'
  - '`llzkOperationIsA_Poly_TemplateParamOp`'
  - '`llzkPoly_TemplateParamOpBuild`'
  - '`llzkPoly_TemplateParamOpGetSymName`'
  - '`llzkPoly_TemplateParamOpSetSymName`'
  - '`llzkPoly_TemplateParamOpGetTypeOpt`'
  - '`llzkPoly_TemplateParamOpSetTypeOpt`'
  - '`llzkOperationIsA_Poly_YieldOp`'
  - '`llzkPoly_YieldOpBuild`'
  - '`llzkPoly_YieldOpGetVal`'
  - '`llzkPoly_YieldOpSetVal`'
  - '`llzkFelt_FeltTypeGetFromRef`'
  - '`llzkFelt_FeltConstAttrGetInField` (use in place of old `llzkFelt_FeltConstAttrGet`)'
  - '`llzkFelt_FeltConstAttrGetWithBitsInField` (use in place of old `llzkFelt_FeltConstAttrGetWithBits`)'
  - '`llzkFelt_FeltConstAttrGetFromStringInField` (use in place of old `llzkFelt_FeltConstAttrGetFromString`)'
  - '`llzkFelt_FeltConstAttrGetFromPartsInField` (use in place of old `llzkFelt_FeltConstAttrGetFromParts`)'

### Changed
- Augment SourceRef API
- Changed field detection logic in IntervalAnalysis
- Clarify `felt.div` in documentation and interval analysis
- FeltConstAttr now directly stores the FeltType rather than field name as a StringAttr
- 'FeltConstAttr syntax changed from `felt.const N <FIELD_NAME>` to `felt.const N : !felt.type<FIELD_NAME>`'
- Rename `CallOp::getCalleeType()` method to `CallOp::getTypeSignature()` to clarify it's not computed from the callee
- Rename `llzk-drop-empty-params` pass to `llzk-drop-empty-templates` and adapt it to new `poly.template` op
- Replace constant parameters on `struct.def` with `poly.template` plus `poly.param` ops
- Subcomponent members (i.e., `struct.member`s of `struct.type`) can no longer be marked as `signal`
- Update `ModuleBuilder` to support `poly.template` and nested module building
- Updated `--llzk-validate-member-writes` pass to correctly handle control flow
- Refactor `llzk::getParentOfType()` to return nullable pointer instead of FailureOr and also check for null input
- Version number updated to 2.0.0
- CAPI:
  - '`llzkStruct_StructDefOpBuild` no longer has `MlirAttribute` parameter'
  - Replaced `llzkStruct_StructDefOpHasConstParamsAttr` with `llzkStruct_StructDefOpHasTemplateSymbolBindings`
  - Renamed `llzkStruct_MemberReadOpBuildWithConstParamDistance` to `llzkStruct_MemberReadOpBuildWithTemplateSymbolDistance`
  - Changed last parameter of `llzkFelt_FeltConstAttrGet` from `MlirIdentifier` to `MlirType`
  - Changed last parameter of `llzkFelt_FeltConstAttrGetWithBits` from `MlirIdentifier` to `MlirType`
  - Changed last parameter of `llzkFelt_FeltConstAttrGetFromString` from `MlirIdentifier` to `MlirType`
  - Changed last parameter of `llzkFelt_FeltConstAttrGetFromParts` from `MlirIdentifier` to `MlirType`
  - Rename `llzkFunction_CallOpGetCalleeType` to `llzkFunction_CallOpGetTypeSignature`

### Fixed
- Bugs causing product program alignment to crash on the circom benchmarks
- Don't build PCL unit tests when PCL is disabled
- Fixed nondeterministic output bug in interval analysis
- Fixed translation of `bool.cmp ne` op to PCL
- Handle `llzk.nondet` ops in `llzk-to-pcl`
- Header files of backends are now installed in the final `include` directory
- Interval analysis lookup bug
- Remove `ConstantLike` and `Pure` traits from `llzk.nondet`
- Support `llzk.nondet` op in SourceRef, Interval lattices
- Treat external `function.call` results as SourceRef roots so accesses through external-call returned values remain trackable in SourceRef-based analyses
- Updated `signal` attribute documentation
- Duplicate symbols sometimes caused assertion failure instead of producing an error diagnostic
- Fix bug that allowed parameterized callee to target a function other than a struct function
- Lower benefit of `GeneralTypeReplacePattern` and similar default patterns to 0 to avoid possible ordering bug, specifically on `FuncDefOp` with pattern from `populateFunctionOpInterfaceTypeConversionPattern()`
- Missing nullptr check in `pod.new` type verifier

### Removed
- '`computeReachable()` and `constrainReachable()` functions from ModuleBuilder'
- meaningless `add_dependencies` from cmake config files
- CAPI:
  - '`llzkStruct_StructDefOpGetConstParams`'
  - '`llzkStruct_StructDefOpSetConstParams`'
  - '`llzkStruct_StructDefOpHasParamName` - use proper `llzkPoly_TemplateOp*` instead'
## v1.1.5 - 2026-03-05
### Fixed
- ensure versioned doc pages are stored in version-specific folders on gh-pages branch

## v1.1.4 - 2026-03-05
### Fixed
- ensure versioned doc pages are stored in version-specific folders on gh-pages branch

## v1.1.3 - 2026-03-05
### Changed
- CI:
  - support required checks on the repo
  - add test result comments to PRs
  - remove job timeouts

### Fixed
- documentation github pages deployment failed on release commits

## v1.1.2 - 2026-03-04
### Fixed
- a couple more issues with setting the version number properly
## v1.1.1 - 2026-03-04
### Fixed
- properly update version number

## v1.1.0 - 2026-03-03
### Added
- CAPI:
  - Added function that generates the PCL lisp representation of a circuit.
### Fixed
- Release workflow
## v1.0.0 - 2026-03-02
### Added
- LLZK IR:
  - core dialects
  - common helper classes and functions for working with LLZK IR
  - unit tests and lit tests
  - documentation pages
- Passes:
  - Analysis:
    - "`llzk-print-call-graph` - Print the call graph"
    - "`llzk-print-call-graph-sccs` - Print Strongly Connected Components in the call graph"
    - "`llzk-print-constraint-dependency-graphs` - Print constraint dependency graph for all LLZK structs"
    - "`llzk-print-interval-analysis` - Print interval analysis results for all LLZK structs"
    - "`llzk-print-symbol-def-tree` - Print symbol definition tree"
    - "`llzk-print-symbol-use-graph` - Print symbol use graph"
    - "`llzk-print-predecessors` - Print the predecessors of all operations"
  - Transformation:
    - "`llzk-array-to-scalar` - Replace arrays with scalar values"
    - "`llzk-inline-includes` - Replace all IncludeOp with contents of the referenced file"
    - "`llzk-drop-empty-params` - Remove empty struct parameter lists"
    - "`llzk-flatten` - Flatten structs (i.e. replace parameterized with instantiated) and unroll loops"
    - "`llzk-duplicate-read-write-elim` - Remove redundant reads and writes"
    - "`llzk-duplicate-op-elim` - Remove redundant arith operations"
    - "`llzk-unused-declaration-elim` - Remove unused member and struct declarations"
    - "`llzk-poly-lowering-pass` - Lowers the degree of all polynomial equations to a specified maximum"
    - "`llzk-inline-structs` - Inline nested structs (i.e., subcomponents)"
    - "`llzk-compute-constrain-to-product` - Replace @compute/@constrain with @product function"
    - "`llzk-fuse-product-loops` - Fuse matching witness/constraint loops in a @product function"
  - Conversion:
    - "`llzk-to-pcl` - Rewrite constraints to be compatible with PCL constraints"
    - "`llzk-r1cs-lowering` - Rewrite constraints to be compatible with R1CS constraints"
    - "`convert-llzk-to-zklean` - Convert LLZK structs and constraints to ZKLean IR"
    - "`convert-zklean-to-llzk` - Convert ZKLean IR back to LLZK structs and constraints"
    - "`zklean-pretty-print` - Pretty-print zkLean dialect IR as Lean code"
  - Validation:
    - "`llzk-validate-member-writes` - Detect multiple and missing writes to the same member of a component"
- llzk-tblgen tool that generates C API for llzk library and linking unit tests for it
- CAPI for LLZK IR and passes
- Backends:
  - r1cs
  - zklean
  - pcl
