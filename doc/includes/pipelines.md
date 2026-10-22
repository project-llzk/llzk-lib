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

_Lower polynomial constraints to R1CS using full polynomial lowering._

Runs the following passes in order:

1. `-llzk-full-poly-lowering` with `lowering={max-degree=2}`
2. `-llzk-r1cs-lowering`
3. MLIR's `-cse`

This existing pipeline retains its original composition. Its conversion stage
uses the updated layout-based R1CS lowering pass.

### `-llzk-constraint-flattening-r1cs-lowering`

_Monomorphize and flatten main constraints, lower degree, and convert layout signals to R1CS._

Runs the following passes in order:

1. `-llzk-constraint-flattening`
2. `-llzk-poly-lowering-pass` with `main-only=true max-degree=2`
3. `-llzk-r1cs-lowering`
4. MLIR's `-cse`

Degree lowering processes the flattened main constrain body and marks generated
auxiliaries as signals. The existing R1CS conversion pass validates degree at
most two, prepares additional R1CS auxiliaries, builds the signal layout, and
converts storage paths directly to wires. Array, POD, and component storage
remain intact until conversion.

Both R1CS pipelines accept `output-llzk=<file>` and `output-layout=<file>`.
Alongside the R1CS IR returned through `-o`, these write the LLZK with auxiliary
fields and assignments in `@compute`, and the access-path/layout-ID/R1CS-ID CSV.
See the R1CS backend documentation for an example.

### `-llzk-constraint-flattening`

_Monomorphize templates and expand main constrain while retaining component storage._

Runs these passes in order:

1. MLIR's `-canonicalize`
2. `-llzk-monomorphize-templates`
3. `-llzk-expose-child-members`
4. MLIR's `-canonicalize`
5. `-llzk-flatten-constraints`

Inline includes and lower symbolic non-native constraint arithmetic before this
pipeline. Constant non-native arithmetic is evaluated during expansion using
the current call and loop bindings. Specialization discovery and constraint
expansion share scalar folding and integer/field conversion logic, so known
template and loop values use the same arithmetic semantics in both stages.
Canonicalization removes frontend `scf.execute_region` wrappers before
monomorphization and folds specialized constants before expansion.
Control flow and all array indices must resolve statically;
aggregate function inputs are read-only. The pass expands concrete constrain
bodies and eliminates local aggregate storage while retaining the original
component layout and ordinary scalar arithmetic. Family handles retain their
rolled types, with concrete member read types validated against the selected
specialization and exposed template declarations. Consumers must support these
reads through rolled handles; `-llzk-array-to-scalar` currently does not.

Only scalar felt equality constraints are supported. Array equality,
`constrain.in`, live non-signal storage reads, recursion, and undefined or
multi-block callees are rejected. See `-llzk-flatten-constraints` for the binding
model, diagnostics, and supported control flow.
