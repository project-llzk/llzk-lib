# In-memory SHA R1CS conversion — 2026-09-27

The new `llzk-translate --llzk-to-r1cs` translation runs monomorphization, symbolic constraint evaluation, degree/R1CS lowering, and binary export in one process. No intermediate IR is printed or reparsed. Rolled compute and witness metadata remain in the module during lowering.

```sh
build/bin/llzk-translate input.llzk --llzk-to-r1cs \
  --r1cs-prime=21888242871839275222246405745257275088548364400416034343698204186575808495617 \
  -o circuit.r1cs
```

## Measurements

Three serial runs per circuit, using the freshly built Release (`-O3 -DNDEBUG`) tools. Seconds exclude the Circom frontend. No tests, builds, or other benchmark jobs were launched alongside the sweep. Old values are the previous three-run medians from [the 29-circuit sweep](r1cs-conversion-timings.md). The old export times were variable, so speedups describe these measurements rather than a guaranteed ratio.

The frontend emits different ordering on repeated invocations. Each new benchmark therefore fixes one generated LLZK input for all three conversion runs and the old-path control. Input hashes are retained in the raw data. Frontend times were measured separately on fresh frontend invocations; the raw end-to-end figures sum those frontend times and fixed-input conversion times.

| Benchmark | Constraints | Previous two-tool median s | In-memory median s | In-memory range s | Speedup vs previous | Fresh old-path control s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sha256_test448 | 408,640 | 76.737 | 39.102 | 37.193–40.150 | 1.96× | 72.788 |
| sha256_test512 | 408,640 | 95.116 | 39.686 | 39.168–39.938 | 2.40× | 129.040 |
| sha256_2_test | 204,465 | 40.519 | 18.361 | 18.243–18.910 | 2.21× | 32.267 |

## Validation

All three in-memory repetitions produced the same binary for each fixed input. Each binary also matched a fresh two-tool conversion byte for byte, including all constraints and wire ordering. The additional old-path controls are single samples, separate from the new medians.

Build and full CMake `check` passed in the Nix release environment: 466 lit tests (four unsupported, one expected failure), plus all 1,342 unit/C API tests. The nested-struct and nested-array regression fixtures now compare the integrated translation against the existing pipeline for both raw and already evaluated input.

[CSV](r1cs-inmemory-sha.csv) · [Raw samples and hashes](r1cs-inmemory-sha.json)

## Native Circom comparison

Fresh native `circom --O0 --r1cs` runs (three serial trials per case) end at binary R1CS and match the LLZK constraint and wire counts. Native times include the native frontend; the end-to-end LLZK column includes the separately measured concrete frontend plus in-memory conversion.

| Benchmark | Native median s | LLZK conversion s | LLZK frontend + conversion s | End-to-end ratio |
| --- | ---: | ---: | ---: | ---: |
| sha256_test448 | 1.555 | 39.102 | 41.370 | 26.60× |
| sha256_test512 | 1.515 | 39.686 | 41.983 | 27.71× |
| sha256_2_test | 1.141 | 18.361 | 19.981 | 17.51× |

[Native samples and comparison](r1cs-native-sha-comparison.json).

## Remaining-cost profile

A macOS sampling profile of the integrated path on the fixed `sha256_2_test`
input identified the following approximate shares of sampled main-thread time.
These are inclusive phase samples, not precise CPU-time measurements or medians;
verifier samples include waiting for parallel verification workers.

| Work | Approximate share |
| --- | ---: |
| Pass-manager verification after passes | 34% |
| Export model construction and binary serialization | 30% |
| Symbolic constraint evaluation | 15% |
| R1CS lowering implementation | 9% |
| CSE implementation | 6% |
| Degree-preparation pipeline (including its nested overhead) | 4% |

The main thread spent 5,273 of 15,369 samples in post-pass verification, and
4,645 samples in the export branch of `lowerAndExportR1CS`. Symbol-table
verification and generic operation/trait traversal were prominent worker stacks.
The earlier pass-timing report therefore must not be interpreted as timing only
the transformation algorithms: pass-manager verification is a substantial
separate cost in this integrated profile.

The exporter builds a second sparse representation from the R1CS operation graph.
`CircuitExportModelBuilder::flattenLinear` memoizes flattened subexpressions,
copies and merges coefficient collections, reduces coefficients, and sorts terms.
The lowering side creates `r1cs.to_linear`/`r1cs.add`/`r1cs.mul_const` operations
and subsequently runs CSE. This build/simplify/flatten cycle is a concrete source
of overhead. Binary coefficient serialization additionally converts DynamicAPInt
values through decimal strings in `toExactWidthAPInt`; that was visible in the
profile but is only part of export cost.

The first optimization experiments should separate necessary boundary verification
from repeated intermediate verification, then reduce the build/simplify/flatten
cycle for linear combinations. This investigation did not disable verification
or change compiler behavior. No measured speedup for either proposal is claimed.
