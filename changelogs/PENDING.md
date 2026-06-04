## v2.1.0 - 2026-06-04
### Added
- Added `verif` dialect:
  - Added `verif.contract` for defining specs on structs and functions
  - Added `verif.include` to invoke contracts from within contracts
  - Added `verif.require_compute` and `verif.require_constrain` for encoding preconditions
  - Added `verif.ensure_compute` and `verif.ensure_constrain` for encoding postconditions
- CAPI:
  - '`llzkSymbolTableInsert` to support inserting a child symbol op with automatic renaming to avoid name collisions'
- CAPI:
  - llzkFunction_CallOpBuildWithTemplateParams
  - llzkFunction_CallOpBuildToCalleeWithTemplateParams
- CAPI:
  - Methods for working with `MlirOpBuilder`
- Configuration:
  - field: the prime field to use for the lowering
- Add `function.arg_name` attribute for tracking source function argument names
- Add `ram` dialect with `ram.load` and `ram.store` ops for witness-generation memory
- Add getters and builders for TemplateParamOp and TemplateExprOp in TemplateBuilder
- Add verifier support for `function.res_name` result attributes.
- An optimized SMT encoding for non-native theories like QF_NIA (and eventually QF_BV)
- Canonicalization for `cast.tofelt(arith.constant n)`
- Canonicalization pattern for cast.toindex ops of constant felts
- Convenience accessors for `FeltConstantOp` values
- Custom `verif.contract` builder that accepted a SymbolRefAttribute for the target, instead of just an identifier string
- Lowering from LLZK to SMT dialect
- Overflow semantics attribute to `cast.toindex` and `cast.tofelt` ops
- Wildcard '?' option for template parameters to function.call
- '`AlwaysSpeculatable` trait on `llzk.nondet` op'
- '`Pure` trait on `poly.read_const` op'
- '`SpecializedSROA` and `SpecializedMem2Reg` passes'
- '`toDynamicAPInt(size_t)` helper function'
- canonicalizer for `llzk.nondet` op to remove it when its result is not used
- llzk-while-to-for pass converting while loops to for loops when possible
- llzk-witgen tool
- more test cases from circom translations

### Changed
- Added `cast` to legal dialects for array-to-scalar pass
- Changed the SourceRefAnalysis to be a sparse analysis instead of a dense analysis
- Document `felt.shl` and `felt.shr` semantics.
- Extending the interval analysis to track unreduced intervals.
- Made interval analysis propagate intervals through concrete subcomponent constrain calls.
- Relax `NotFieldNative` trait verification to allow non-native field ops in `poly.expr` parents.
- Replace nondet array allocation with array.new in ArrayToScalar pass
- The `verif.contract` op no longer must be located within the same template and/or module as the target struct/function
- The old encoding is called a naive encoding where modulos are always applied during an integer encoding.
- Update ArrayToScalarPass to use these new passes instead of the builtin ones

### Fixed
- CAPI: Fix incorrect behavior in `llzkPod_NewPodOpBuildWithMapOperands` constructor.
- CAPI:
  - Memory errors while handling insertion points
- Added support for smod in the interval analysis
- Avoid assertion failure in `ArrayIndexGen::linearize` and `ArrayIndexGen::checkAndConvert` for constant integer indices larger than int64_t max
- Avoid unnecessary narrowing to `int` in `Field::reduce()`
- Bounds-check constant `array.len` dimensions before array-to-scalar lowering
- Ensure distribution of random numbers is uniform for large primes (>64 bits)
- Expose custom `ContractOp` builder to the CAPI
- Felt constant used as parameter to StructType was incorrectly treated as invalid
- Fix R1CS lowering for quadratic-equals-linear constraints so the emitted constraint preserves the linear side's sign.
- Fixed `DynamicAPInt` shift operations with shift amounts larger than 64 bits
- Fixed `llzk-duplicate-op-elim` so it does not merge distinct `llzk.nondet` witness values.
- Fixed `llzk-flatten` when instantiated struct bodies call through type parameters
- Fixed imprecision of interval analysis during member writes of array values
- Fixed incorrect behavior in `pod.new` constructor that accepts both record initialization operands and map operands.
- Fixed interval analysis for storage writes under control-dependent regions
- Fixed mod operation handling
- Fixes issue 502 where SMT lowering occasionally triggers segfaults.
- In llzk-array-to-scalar the unwrapped members switched the {column} and {signal} attributes
- Scalarize array reads inside branches when a matching array write dominates the branch.
- The `llzk.main` attribute was not always verified
- Update legalization for empty template removal for new dialects/types/ops
- '`toDynamicAPInt(APSInt)` incorrectly produced negative results for values larger than MAX(int64_t) and up to 64 bits'
- allow prover-side recursion outside the `@constrain`-reachable slice in the inline-structs pass
- assertion failure in `remove-dead-values` pass when there are empty `else` regions
- bug in `llzk-tblgen` enum C API tests for non-uppercase enum case symbols
- bug in `llzk-tblgen` that used a specific type instead of general Type/Attribute class
- bug in `llzk-tblgen` where generated setter did not handle VariadicOfVariadic operand groups correctly
- exclude `pod.type` from valid types for `poly.read_const` op
- fix bug in llzk-tblgen where the generated getter/setter for a variadic operand was using the wrong offset when there are multiple variadic operands.
- fix invalid FileCheck lines generated by generate-test-checks.py
- fix tblgen bug involving non-variadic offsets when there are variadic operands
- outdated documentation in TypeHelper

### Removed
- '`Pure` trait from `poly.expr` operation (allows dead value removal to be used on `poly.expr`)'

