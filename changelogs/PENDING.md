## v2.1.2 - 2026-06-18
### Added
- AI coding agent instructions file
- Add `--llzk-remove-unused-discardable-allocations` for removing unread discardable allocations and their dead stores.
- Destructurable*Interface and Promotable*Interface to pod ops/types
- PodRefOpInterface and PodAccessOpInterface for ops that reference or access pods
- '`--check-output` option for `llzk-witgen`'
- '`llzk-pod-to-scalar` pass to destructure pod type values into scalar SSA values'
- '`llzk-specialize-wildcard-arrays` pass to refine array types with wildcard dimensions'

### Changed
- Allow non-native field ops in `verif.contract`
- Apply `llzk-flatten` cleanup option to free functions, not just structs
- '`verif.contract` for the `llzk.main` struct rejects direct `verif.require_*` ops as well'
- '`verif.contract` now rejects preconditions (`verif.require_*`) derived from struct members or function return values'

### Fixed
- Added `scf.if` handling for `llzk-witgen --backend=execution-engine`
- Emit all R1CS-lowering auxiliary members with the exact type of the materialized expression
- Emit polynomial-lowering auxiliary members with the exact type of the materialized expression
- Emit synthesized zero R1CS linear-combination constants with a printable integer width
- Handle wildcard CallOp template parameters in the flattening pass.
- Implement missing felt ops for `llzk-witgen`
- Lower high-degree felt.add, felt.sub, and felt.neg equality roots and nonlinear struct constrain call arguments in the poly-lowering pass
- Prevent `--cse` from merging distinct `array.new` allocations.
- Prevent `--cse` from merging distinct `pod.new` allocations.
- Reject polynomial and R1CS lowering on non-straight-line constrain bodies instead of materializing component-scope auxiliaries inside control flow
- Update SourceRefAnalysis to handle `scf.while`, `verif.contract`, and `verif.include`
- array and pod scalarization passes used the original RDV pass instead of the wrapper with bug fix

