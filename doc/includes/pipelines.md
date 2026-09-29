### `-llzk-remove-unnecessary-ops`

_Remove unnecessary operations, such as redundant reads or repeated constraints._

Runs the following cleanup passes in order:

1. `-llzk-duplicate-read-write-elim`
2. `-llzk-duplicate-op-elim`

### `-llzk-remove-unnecessary-ops-and-defs`

_Remove unnecessary operations and member definitions._

Runs `-llzk-remove-unnecessary-ops`, followed by:

3. `-llzk-unused-declaration-elim`

The unused-declaration pass uses its default options, so it removes unused
members but does not remove struct definitions.

### `-llzk-product-program`

_Convert `@compute`/`@constrain` functions to an `@product` function and perform alignment._

Runs the following passes in order:

1. `-llzk-compute-constrain-to-product`
2. `-llzk-fuse-product-control-flow`

### `-llzk-full-struct-inlining`

_Flatten and inline all struct definitions into the main struct._

Runs the following passes in order:

1. `-llzk-flatten`
2. `-llzk-pod-to-scalar`, when `pod-to-scalar` is enabled
3. `-llzk-array-to-scalar`, when `array-to-scalar` is enabled
4. MLIR's `-canonicalize`
5. `-llzk-inline-structs`
6. `-llzk-unused-declaration-elim` with `remove-structs=true`

POD scalarization precedes array scalarization because it can split
`pod.type` array elements into parallel arrays. The pipeline defaults
flattening cleanup to `main-as-root`; using another cleanup mode is not
recommended because parameterized templates may survive flattening and cause
`-llzk-inline-structs` to fail.

#### Options

- `flattening={max-iter=<n>}` sets the maximum number of flattening iterations
  (default: `1000`).
- `flattening={cleanup=<mode>}` selects the flattening cleanup mode (default:
  `main-as-root` for this pipeline).
- `array-to-scalar` controls whether `-llzk-array-to-scalar` runs
  (default: `true`).
- `pod-to-scalar` controls whether `-llzk-pod-to-scalar` runs
  (default: `true`).
- `inlining={max-merge-complexity=<n>}` limits the complexity of merged structs
  (default: `0`, meaning no limit).

### `-llzk-full-inlining`

_Inline free functions, then flatten and inline struct definitions._

This is the recommended pipeline before a downstream pass that does not
support `function.call`. It first runs `-llzk-inline-free-functions`, then
runs the same sequence and accepts the same options as
`-llzk-full-struct-inlining`.

### `-llzk-full-poly-lowering`

_Flatten and inline structs, lower polynomial constraints, then remove unused IR._

Runs the following passes in order:

1. The complete `-llzk-full-struct-inlining` sequence
2. `-llzk-poly-lowering-pass`
3. The complete `-llzk-remove-unnecessary-ops-and-defs` sequence

The nested struct-inlining stage uses `main-as-root` cleanup by default for
the same reason as `-llzk-full-struct-inlining`.

#### Options

- `flatten-inline` configures the nested struct-inlining stage. Its nested
  `flattening`, `array-to-scalar`, `pod-to-scalar`, and `inlining` options
  have the same meaning and defaults as the corresponding
  `-llzk-full-struct-inlining` options.
- `lowering={max-degree=<n>}` sets the maximum polynomial degree accepted by
  `-llzk-poly-lowering-pass` (default: `2`; minimum: `2`).

### `-llzk-full-r1cs-lowering`

_Lower evaluated constraints to R1CS, or opt into legacy flattened lowering._

By default, input must have been processed by `-llzk-monomorphize` and
`-llzk-evaluate-constraints`, retaining its original storage interface.
Runs degree lowering to degree two, `-llzk-r1cs-prepare`,
`-llzk-r1cs-lowering`, and MLIR's `-cse`.

Use `-llzk-full-r1cs-lowering="legacy=true"` for the legacy pipeline: full
polynomial lowering (including flattening), R1CS preparation, legacy emission,
and CSE. Legacy mode rejects evaluated input. There are no separate direct
pass or pipeline names.
