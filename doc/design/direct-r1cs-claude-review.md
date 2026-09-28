# Follow-up to Claude's direct R1CS review

Reviewed against `d30ba1292` on 2026-09-28. The numbered rows match the supplied findings.

| # | Assessment and resolution |
|---|---|
| 1 | Confirmed. Array index constants are now created only on storage-prefix cache misses. Added a small two-leaf regression and regenerated the affected family/grid checks. |
| 2 | Confirmed performance cost. Aggregate wrappers now share immutable contents and detach on writes. Reads and argument copies no longer recursively copy arrays. Existing copy-boundary tests cover value isolation; a unit test verifies storage sharing and detachment. |
| 3 | A design invariant, not a demonstrated disclosure. Preserve the requested public nested-field access model, document the distinction, and make witness visibility consumers consult `poly.original_public`. Public serialization no longer depends solely on the main-output type restriction. |
| 4 | Confirmed avoidable cache flushing. Replace wholesale clearing with bounded LRU eviction. The capacity test queries a retained entry and an evicted entry. This does not promise hits for a cyclic working set larger than the cache. |
| 5 | Split into registered preparation, legacy emission, and direct emission passes. Remove the dynamic wrapper; expose explicit Legacy/Direct pipeline modes and two flat registered pipelines. Nested MLIR instrumentation already worked, but the static pass lists now also make the input contract explicit. |
| 6 | Confirmed. A dedicated `llzk-r1cs-prepare` pass establishes `r1cs.prepared` after shared normalization for both paths. The new two-process regression also exposed missing degree memo reconstruction in legacy export; rebuild those degrees before lowering equations. |
| 7 | Defensive invariant. Assert matching visibility when inserting another value for an existing storage path. The reviewer did not identify a currently reachable disagreement. |
| 8 | Add missing compute and matching-constrain-arity guards in evaluated witness collection. Ordinary driver and lowering paths already reject missing functions; the local guard removes reliance on those upstream checks. |
| 9 | Confirmed inconsistency in predicates, without a demonstrated normal-pipeline failure. Witness selection, serialization, preparation, and WTNS dispatch now use the authoritative module marker through one helper. Add a binding-selection test without a function marker. |
| 10 | Confirmed cleanup. Collect the witness once after lowering; obtain only the relevant circuit name, explicitly validating the evaluated circuit reference. Remove the throwaway successful Expected and duplicate evaluated branches. |
| 11 | Multiword exponent coverage was missing. Add Fermat's inverse exponent and a deterministic random full-width exponent for bn128. Zero bases with nonzero exponents were already covered by the existing oracle inputs. |

## Aggregate read microbenchmark

Each run initializes an n-element field array in a struct, reads that member once per loop iteration, and sums its scalar elements. Both versions produced the same checked output. These are process-level medians of three runs, including startup and JSON parsing, using the interpreter; they are not end-to-end circuit compilation measurements.

| Elements | Before | Copy-on-write |
|---|---:|---:|
| 2,048 | 47.6 ms | 10.9 ms |
| 4,096 | 135.7 ms | 16.5 ms |

A value copy creates a fresh aggregate wrapper that shares its contents. SSA aliases retain the same wrapper. Writes detach the container when shared; nested aggregate reads also create fresh wrappers. Thus read-only copies are constant-time while a first write can still cost the size of its container.

## Validation

`nix develop .#release --offline --command bash -c 'cmake --build build --target check -j 8'` completed successfully: 513 lit tests passed (4 unsupported, 1 expected failure), and all 1,353 unit/C API tests passed. Changed C++ files were clang-formatted. FileCheck baselines were generated with `scripts/generate-test-checks.py` into temporary files and inspected before applying them.
