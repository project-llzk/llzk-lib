# Expanded symbolic evaluator benchmarks

Ran 29 additional circuits from the Circomlib test corpus in both templated and concrete
frontend modes: 58 LLZK attempts, plus 29 fresh native Circom O0 baselines. Runs were
serial (`--jobs 1`), with a 60-second limit for each frontend/llzk-opt process.
No compiler implementation was changed for this sweep. No PoseidonEx case was run.

- Templated: 13 passed; 16 failed in frontend translation.
- Concrete: 19 passed; eight rejected dynamic conditionals; two timed out during evaluation.
- Union: **20 of 29 circuits passed in at least one mode**.
- All 32 successful mode/circuit runs matched native O0 equation counts exactly.
- All 32 also matched native wire counts after adding the constant wire to the evaluator signal count.
- All 29 native O0 compilations succeeded.

These checks establish successful interpretation, output IR verification, and matching
counts. This sweep did not perform witness/constraint-equivalence checks on the new
circuits. The numerical witness checks in the preceding report apply to Poseidon3/6,
not automatically to these circuits.

## Successful circuits

Concrete mode is shown where it succeeds; IsEqual uses templated mode. Full results
for both modes, counters, memory, native baselines, and failure diagnostics are in
[the CSV](symbolic-evaluation-expanded.csv).

| Circuit | Mode | Equations | Eval ms | Frontend s | LLZK process s | LLZK peak MiB | Native O0 s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| isequal | templated | 4 | 0.38 | 0.187 | 0.390 | 23.34 | 0.128 |
| greaterthan | concrete | 39 | 0.78 | 0.038 | 0.020 | 23.81 | 0.040 |
| mux1_1 | concrete | 14 | 0.25 | 0.038 | 0.020 | 23.98 | 0.036 |
| mux2_1 | concrete | 28 | 0.39 | 0.039 | 0.020 | 24.14 | 0.040 |
| mux3_1 | concrete | 47 | 0.61 | 0.040 | 0.020 | 24.56 | 0.040 |
| babyadd_tester | concrete | 6 | 0.15 | 0.039 | 0.020 | 22.53 | 0.037 |
| babycheck_test | concrete | 3 | 0.12 | 0.037 | 0.020 | 22.47 | 0.076 |
| edwards2montgomery | concrete | 2 | 0.11 | 0.020 | 0.021 | 22.50 | 0.020 |
| montgomery2edwards | concrete | 2 | 0.12 | 0.019 | 0.020 | 22.59 | 0.020 |
| montgomeryadd | concrete | 3 | 0.13 | 0.020 | 0.020 | 22.67 | 0.020 |
| montgomerydouble | concrete | 4 | 0.15 | 0.019 | 0.020 | 22.67 | 0.019 |
| sum_test | concrete | 200 | 3.54 | 0.036 | 0.020 | 24.56 | 0.038 |
| sign_test | concrete | 521 | 121.61 | 0.073 | 0.185 | 27.17 | 0.073 |
| constants_test | concrete | 33 | 0.70 | 0.038 | 0.020 | 23.58 | 0.039 |
| babypbk_test | concrete | 10,114 | 157.31 | 0.182 | 0.296 | 62.67 | 0.129 |
| escalarmulfix_test | concrete | 10,114 | 156.96 | 0.187 | 0.290 | 62.62 | 0.128 |
| pedersen2_test | concrete | 8,127 | 133.85 | 0.239 | 0.278 | 54.97 | 0.128 |
| sha256_test448 | concrete | 408,640 | 5941.27 | 2.499 | 18.910 | 713.98 | 1.826 |
| sha256_test512 | concrete | 408,640 | 5118.82 | 2.449 | 16.092 | 925.98 | 1.767 |
| sha256_2_test | concrete | 204,465 | 2588.18 | 1.700 | 8.158 | 521.72 | 1.163 |

Evaluator time excludes parsing, specialization, and pass-manager verification.
LLZK process time and peak memory include all those costs. Native O0 time includes
compilation and R1CS writing. LLZK output is discarded after verification and has not
been lowered to R1CS; these are different compilation endpoints, not end-to-end
performance equivalents. Each number is one sample. No build ran during measurement.

## Observations and failures

Fixed-base scalar multiplication and public-key derivation each produce 10,114 equations
in about 157 ms of evaluator time. Pedersen2 produces 8,127 equations in 134 ms.
The SHA cases reach 204,465–408,640 equations, taking 2.6–5.9 seconds for evaluation
and 8.2–18.9 seconds for the full LLZK process. Native O0 compilation is faster on
SHA (1.2–1.8 seconds), so the earlier Poseidon timing advantage does not generalize.
Peak LLZK process memory reaches 926 MiB on SHA-256/512.

Concrete dynamic-condition failures occur in `isequal`, `pointbits_loopback`,
`escalarmulany_test`, all three EdDSA cases, and both depth-10 sparse Merkle cases.
Notably, templated IsEqual succeeds; this is a frontend-representation-dependent
limitation rather than an unconditional inability to represent equality constraints.

`escalarmul_min_test` and `pedersen_test` finish specialization (70 and 135
definitions respectively) but exceed the 60-second optimizer limit during evaluation.
Their native O0 counts are only 6,150 and 13,108 equations; output size alone does not
explain these timeouts. Profiling the interpreter on these two cases is a useful next
step. No evaluator completion counters or reliable timeout peak RSS are reported.

The 16 templated frontend failures include missing helper definitions (`Num2Bits`,
`IsZero`, `H`, `Segment`, `AND`), missing `hin`/`nInputs` bindings, unsupported affine
dimensions, BinSum expression/type errors, and array-valued polymorphic constants.
The concrete frontend translated all 29 inputs successfully.

## Reproduction

Used the previously validated Release package:
`/nix/store/bwj2iplnm0f626h5b6f3lasz0xxc9jyb-llzk-release-3.0.0/bin/llzk-opt`.
Frontend: Circom 2.2.2, checkout `bf3b28ea35f7530c6ce3e60c77532f5f9b9f69a7`.
LLZK base: `52a7466b554558088076735387c293db9426725c` plus the recovered checkpoint
and evaluator implementation described in [the implementation report](symbolic-constraint-evaluation.md).

```sh
nix develop .#release --command bash -c 'python3 scripts/benchmark-monomorphization.py \
  --manifest scripts/symbolic-evaluation-benchmarks.json \
  --corpus /Users/shankarapailoor/veridise/circom-benchmarks \
  --frontend /Users/shankarapailoor/veridise/circom/result/bin/circom \
  --llzk-opt /nix/store/bwj2iplnm0f626h5b6f3lasz0xxc9jyb-llzk-release-3.0.0/bin/llzk-opt \
  --phase evaluate --frontend-mode concrete --plaintext --jobs 1 \
  --build-type Release --timeout 60 --output /tmp/llzk-expanded-concrete'
```

Repeat with `--frontend-mode templated` and a different output directory. Native
baselines use the same source files with `circom <source> --O0 --r1cs -o <temporary>`
and `/usr/bin/time -l` on macOS. Temporary emitted IR, native R1CS files, and logs
were removed after summarization. The checked-in CSV contains only compact metrics
and diagnostics. No C++/TableGen/build files changed, so the existing validated
Release build was reused without rebuilding.

## Follow-up: conditional-failure investigation

The eight concrete-mode conditional diagnostics do not establish that conditional
constraint semantics are needed. Reproduction with source locations traced seven
of them to `comparators.circom:30`: `inv <-- in!=0 ? 1/in : 0`. Its witness-dependent
condition survives in `IsZero.constrain`, but both branches are empty. The
evaluator unnecessarily demands a known condition before noticing this.

Deleting only those empty branches in temporary frontend IR (without running
canonicalization) produced these successful results:

| Circuit | Constraints | Evaluator milliseconds |
| --- | ---: | ---: |
| isequal | 4 | 0.58 |
| escalarmulany_test | 8,131 | 118.86 |
| eddsamimc_test | 21,737 | 550.29 |
| eddsaposeidon_test | 21,246 | 579.24 |
| smtverifier10_test | 12,582 | 738.77 |
| smtprocessor10_test | 20,465 | 1,129.07 |

All six equation counts match the native O0 baseline. These are single diagnostic
runs on modified temporary IR, not updated benchmark results or equivalence proofs.

`eddsa_test` then reaches the same blocker as `pointbits_loopback`: the `sqrt`
helper at `pointbits.circom:29`. The source computes `x = sqrt(...)`, adjusts its
sign, and uses it in the witness-only assignment `out[0] <-- x`. The constrain
body retains this computation even though constraints read the stored `out[0]`
signal instead. Canonicalization removes the unused sign branch but retains the
unused function call. Removing that specific unused call from a temporary
canonicalized pointbits module allows evaluation to finish with 5,687 equations,
matching native O0 (995.94 ms). End-to-end success for eddsa after this cleanup has
not been established. A general fix must account for callee effects rather than
discarding arbitrary unused calls.

Canonicalization is not a sufficient workaround: it converts static `scf.if`
expressions for segment sizes into `arith.select`. The evaluator accepts only
attribute-valued fold results, ignoring folds that select an existing SSA value.
Consequently the static segment bound becomes symbolic, producing a while-bound
failure in scalar multiplication and signature circuits. Keeping their original
static branches avoids that failure, as the table above demonstrates.

The implementation needs safe handling of empty branches and dead witness-only
computations, plus propagation of value-valued folds (or explicit static select
evaluation). No compiler implementation was changed during this investigation.

## Implemented cleanup

The follow-up implementation based on `ad76a9f1389692b3f5a64c9ba1e9afc18f1cb257`
cleans a temporary module and appends only the successfully generated flat function
to the original. Rolled templates, specialized compute/constrain definitions, and
source operation ordinals are preserved. Cleanup uses folding, dead-code removal,
`scf.if` canonicalization, and removal of unused calls with transitively proven
absence of effects. It does not inline or unroll source bodies. Constraints,
assertions, mutations, unknown operations/callees, external declarations, and
recursive callees are retained conservatively. SSA-valued folds now propagate
abstract values, fixing static selects introduced by branch simplification.

All **27 concrete cases** other than the two pre-existing timeouts now complete
evaluation and output verification. All 27 match freshly regenerated Circom O0
equation counts and wire counts (referenced signals plus the constant wire).
The eight formerly failing cases are:

| Circuit | Emitted equations = native O0 constraints |
| --- | ---: |
| isequal | 4 |
| pointbits_loopback | 5,687 |
| escalarmulany_test | 8,131 |
| eddsa_test | 45,259 |
| eddsamimc_test | 21,737 |
| eddsaposeidon_test | 21,246 |
| smtverifier10_test | 12,582 |
| smtprocessor10_test | 20,465 |

[All count results](symbolic-evaluation-cleanup.csv) also cover the 19 previously
successful concrete cases, including fixed-base multiplication, Pedersen2, and
all three SHA cases. These are count checks and verified IR, **not witness
equivalence checks**. No new witness-equivalence claim is made for EdDSA or the
other newly passing circuits. No frontend repository changes were needed.

Reproduction uses the command above with `--llzk-opt build/bin/llzk-opt`,
`--filter '^(?!(escalarmul_min_test|pedersen_test)$).*'`, and a fresh output
directory. Configure the local Release build with `-DLLZK_VERSION_OVERRIDE=3.0.0`
to match the Nix package's bytecode version. Native references were regenerated
with `--O0 --r1cs`. Timing comparisons are deliberately omitted: portions of this
validation overlapped local build/test activity.

The two prior timeouts were not rerun or optimized. Observable dynamic control
flow remains unsupported. The effect proof intentionally keeps unused helpers
that mutate even local arrays or have unclassified effects; these may still
require future escape/effect analysis. Templated frontend translation limitations
are unchanged and this follow-up did not repeat the templated sweep.

Focused regressions cover empty branches, dead transitive helper calls, an unused
sign calculation, retained constraint calls, assertions, conditional array writes,
recursion, external declarations, and static selects with call-frame constants.
A text comparison checks complete retained source preservation. The existing
dynamic-loop rejection fixture now contains an equation, since its previous
empty body is legitimately removed by dead-code cleanup. FileCheck patterns were
generated only after a successful non-inplace dry run.

Final validation: `nix build -L` succeeded, producing
`/nix/store/wwg3gwviyic54bxgcprf46zikmlbvdfz-llzk-release-3.0.0`.
An offline invocation built the same derivation while the original invocation
retried an unavailable binary-cache hostname; both completed successfully.
The package passed **461 lit tests** (four unsupported, one expected failure) and
**all 1,340 unit/C API tests**. The standalone Release `check-lit` target also
passed. No build-affecting changes followed this validation.

## Timeout profiling after cleanup

Reran `escalarmul_min_test` and `pedersen_test` in concrete mode with the validated
cleanup build, using the same 60-second per-process limit. Both still time out
in evaluation after successful specialization. Two-second macOS `sample` profiles
show active constant arithmetic, rather than a deadlock or equation-output growth:

| Inclusive sampled stack | EscalarMul | Pedersen |
| --- | ---: | ---: |
| Main-thread samples | 1,528 | 1,690 |
| `Operation::fold` | 1,522 (99.6%) | 1,681 (99.5%) |
| `toDynamicAPInt(const APInt&)` | 1,203 (78.7%) | 1,352 (80.0%) |
| `modInversePrime` | 290 (19.0%) | 297 (17.6%) |

These are short statistical samples, not whole-run timing measurements. Both
circuits use `EscalarMulW4Table`, which computes constant elliptic-curve point
addition/doubling tables. Each window recomputes its doubling prefix; each
`pointAdd` contains two field divisions. The 256-bit case's source loops imply
9,024 point additions across its 64 windows, despite only 6,150 native equations.
The old Pedersen circuit uses this same table builder for two 250-bit segments.

The immediate hot path is `lib/Util/DynamicAPIntHelper.cpp`:
`toDynamicAPInt(const APSInt&)` constructs large integers one bit at a time using
`DynamicAPInt` multiplication/addition. Field folding converts both operands
through this helper on every binary fold. Division additionally computes an
inverse by Fermat exponentiation. Existing scalar-expression CSE does not cache
these constant fold evaluations.

The first optimization candidate is a faster, exactly equivalent APInt conversion
(e.g. assembling machine-word chunks), with signed/unsigned boundary tests.
Memoizing suitable constant folds or effect-free helper results could additionally
avoid repeated table-prefix work. Neither optimization was implemented during
this bounded investigation; the compiler and previously validated build remain
unchanged. Raw profiles and rerun logs were written under `/tmp`.

## APInt folding experiment

Replaced DynamicAPInt round trips in field add/subtract/multiply, negate,
power, divide, and inverse folding with unsigned APInt arithmetic. Addition and
multiplication first attempt the existing operand width; an overflowing result
is recomputed from the original operands at a sufficient width. Subtraction
handles negative differences in the field. Powers and inverses use exponentiation
by squaring with exact products. The Field object retains its prime as APInt so
these folds also avoid converting the modulus.

The implementation skips unsigned remainder when the result is already below
the prime. Results stored in MLIR attributes remain canonical field values;
reduction is **not deferred across separate IR operations**. This preserves the
existing semantics of comparisons, integer casts, control flow, and emitted
constants. Signed integer division/remainder and bitwise/shift folds retain their
existing DynamicAPInt paths. No inversion, constant-fold, or call cache was added.

Both former timeouts now finish with verified output and counts matching the
native O0 baselines:

| Circuit | Previous optimizer limit | Evaluation seconds | Optimizer wall seconds | Equations | Signals + constant wire |
| --- | ---: | ---: | ---: | ---: | ---: |
| escalarmul_min_test | >60 s | 5.289 | 6.524 | 6,150 | 6,407 |
| pedersen_test | >60 s | 10.294 | 12.244 | 13,108 | 13,109 |

These are single serial Release samples on the same machine, with no concurrent
build or test run during measurement. Optimizer wall time includes parsing,
specialization, cleanup, evaluation, and verification. The previous runs were
terminated, so they establish lower bounds on improvement, not exact speedup
ratios. Counts are not witness-equivalence proofs.

The focused folding suite passed all 47 tests. New oracle coverage compares
APInt results against the existing DynamicAPInt arithmetic for BabyBear,
Goldilocks, and BN128: zero, one, values near the prime, seeded random values,
mixed operand widths, overflowing sums/products, subtraction underflow, powers,
inverses, and division by zero. Canonical output values are checked explicitly.

For comparison, the previously recorded native Circom O0 R1CS compilations took
0.606 seconds (EscalarMul) and 1.028 seconds (Pedersen). LLZK frontend translation
in the new run took 1.920 and 4.746 seconds respectively, giving frontend-plus-
optimizer totals of 8.444 and 16.991 seconds. Native remains substantially faster;
its endpoint also includes R1CS writing, while LLZK stops at verified scalar IR.
These native timings are the earlier recorded baselines, not new simultaneous
measurements. The optimization resolves the timeouts without closing that gap.

The 27 previously passing concrete cases were also rerun successfully. Combined
with the two resolved timeouts, **all 29 concrete cases now pass**, with exact
native O0 equation-count and signal-plus-constant-wire-count matches. The 27-case
regression rerun overlapped build activity, so it is used for correctness counts,
not timing comparisons. The standalone lit suite passed 461 tests, with four
unsupported and one expected failure.

Full validation succeeded with `nix build -L --offline`, producing
`/nix/store/ykxn33q7xqamzpdh0lgj36b6akhwdamh-llzk-release-3.0.0`.
The package passed 461 lit tests (four unsupported, one expected failure) and
all 1,341 unit/C API tests, including the new arithmetic oracle regression.

## Remaining cost after APInt folding

A fresh Pedersen rerun with the optimized build completed in 11.722 seconds
optimizer wall time, including 9.979 seconds in evaluation. A two-second macOS
CPU sample taken during evaluation contained 1,551 main-thread samples:

- 1,457 (93.9%) in `DivFeltOp::fold`;
- 1,437 (92.6%) in its `powerModulo` exponentiation;
- 659 (42.5%) in `exactMultiply` and 708 (45.6%) in `reduceUnsigned`;
- zero samples in the previous `toDynamicAPInt(const APInt&)` conversion.

These are inclusive stack counts: the multiplication/reduction samples are inside
exponentiation and must not be added to its percentage. They describe the sampled
interval, not a whole-process instrumentation profile. Roughly 1.74 seconds of
optimizer wall time lies outside the reported interpreter interval (parsing,
specialization, cleanup, verification, and other overhead combined).

For BN128, `p-2` has 254 bits and 127 set bits. The current square-and-multiply
implementation performs 253 squares plus 127 conditional products per inversion,
each followed by reduction. The source-level constant-table replay counted 35,028
inversions for Pedersen, implying 13,310,640 modular products in those inversions.
Only 4,273 denominators were distinct (87.8% potential inverse-cache hits).
EscalarMul similarly had 18,048 inversions but only 2,171 distinct denominators.
These repetition counts come from replaying the source arithmetic, not instrumented
compiler counters.

With conversion overhead removed, caching inverse results by field modulus and
canonical denominator is now a directly targeted next optimization. Constant-fold
or pure-call memoization could avoid additional repeated work. A faster inversion
algorithm or modular multiplication/reduction could improve remaining cache misses.
No additional implementation change was made for this profiling run.

## Evaluator-local inversion cache

Added a cache owned by each evaluator invocation, keyed by the numeric prime
modulus and reduced denominator with normalized APInt widths. Division and
explicit inversion share entries; cached numeric results are rewrapped in the
requested field type so aliases of one prime can share work without mixing types.
Zero denominators bypass the cache and keep the existing folding behavior.
The cache clears at 8,192 entries, bounding retained entries without changing
results. Arithmetic still uses the existing field-op folds: cache misses fold an
inverse, and divisions multiply the numerator by that inverse. There is no global
or cross-evaluation cache and no new arithmetic implementation.

Serial Release measurements, without concurrent build or test activity:

| Circuit | Previous eval s | Cached eval s | Previous optimizer s | Cached optimizer s | Cache hits | Cache misses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| escalarmul_min_test | 5.289 | 0.977 | 6.524 | 1.825 | 15,877 | 2,171 |
| pedersen_test | 10.294 | 1.970 | 12.244 | 3.983 | 30,755 | 4,273 |

Both retain exactly the native O0 equation and wire counts. Measured cache
hits/misses exactly match the previous source-level replay's predictions.
These single-sample evaluator improvements are about 5.4x and 5.2x; total optimizer
improvements are about 3.6x and 3.1x. This is still a scalar-IR endpoint rather than
native Circom's R1CS endpoint, and count checks do not prove witness equivalence.
The benchmark driver now records the two inverse-cache counters from evaluator
reports.

Regressions check reuse across different numerators, sharing between division and
explicit inversion, isolation of distinct primes, sharing between field aliases,
zero exclusion, and correct recomputation after capacity eviction. The eviction
case checks 8,195 folded identities `x * inverse(x) = 1`. FileCheck patterns for
the small arithmetic case were generated after a successful temporary-output dry
run.

All 29 concrete cases were validated again with caching and match the native O0
equation and wire counts. The standalone lit suite passed 463 tests, with four
unsupported and one expected failure.

A five-second sample launched with the cached Pedersen optimizer captured 3,510
main-thread samples: 1,292 (36.8%) in modular exponentiation for cache misses,
804 (22.9%) in textual source parsing, 152 (4.3%) in greedy cleanup, and 46 (1.3%)
in verification. Exponentiation accounts for roughly half the sampled interpreter
work; other folding, interpretation, and allocation comprise most of the rest.
The sample overlapped the package build, so these approximate stack shares are
not a replacement for the isolated timing measurements above. Remaining targets
are the 4,273 distinct inversions and textual-IR parsing. Faster inversion and
bytecode input are possible follow-ups, neither implemented here.

Frontend translation in the isolated cached runs took 1.864 seconds for EscalarMul
and 4.789 seconds for Pedersen. Including the optimizer gives totals of 3.689 and
8.772 seconds respectively. Frontend translation is therefore now the largest
single stage in this end-to-end Pedersen pipeline.

Final cache validation: `nix build -L --offline` succeeded, producing
`/nix/store/6d06ym96idsqk4j282ln21qcl14xg15c-llzk-release-3.0.0`.
All 463 lit tests and 1,341 unit/C API tests passed (four lit cases unsupported,
one expected failure). No compiler changes followed this build.

## GMP inversion

Replaced Fermat exponentiation in `felt.inv` and `felt.div` constant folding with
GMP `mpz_invert`. APInt operands transfer directly through `mpz_import` and
`mpz_export` using least-significant-word-first uint64_t arrays and native byte
order. The evaluator-local cache remains unchanged. Other arithmetic and explicit
`felt.pow` still use APInt. Non-invertible operands remain unfolded, including
noncanonical multiples of the field prime. GMP is now a required build dependency;
its CMake discovery module is installed for downstream static-library consumers,
and Nix propagates the dependency.

Three serial alternating before/after runs per circuit used identical frontend
IR and Release executables. Medians below include conversion overhead. Unlike the
previous dump-free benchmark, these optimizer wall times include writing the full
output IR so its SHA-256 can be compared. All six outputs for each circuit were
byte-for-byte identical; equation counts, signal counts, and cache counters also
matched. No build or test work ran concurrently with these measurements.

| Circuit | Cached Fermat eval s | Cached GMP eval s | Eval speedup | Fermat optimizer s | GMP optimizer s |
| --- | ---: | ---: | ---: | ---: | ---: |
| escalarmul_min_test | 1.167 | 0.477 | 2.45x | 2.409 | 1.620 |
| pedersen_test | 2.293 | 0.939 | 2.44x | 4.567 | 3.551 |

Frontend compilation, run once per circuit and shared by both variants, took
2.066 and 5.316 seconds respectively. This optimization does not change that
stage. Raw local measurements are in `/tmp/llzk-gmp-comparison.json`.

Adding the shared frontend time to each optimizer median gives end-to-end stage
sums of 4.475 -> 3.686 seconds for EscalarMul (17.6% reduction), and
9.883 -> 8.867 seconds for Pedersen (10.3% reduction). These are stage sums,
not separately repeated end-to-end pipeline measurements. Frontend compilation
accounts for about 60% of the resulting Pedersen total.

The local Release `check` target passed all 463 lit tests and 1,342 unit/C API
tests (four lit cases unsupported and one expected failure). New unit coverage
checks multiword/noncanonical inputs and rejects inversion/division by multiples
of the modulus; the existing exact-arithmetic oracle covers random and boundary
values for babybear, goldilocks, and bn128.

Clean Nix package and downstream consumer validation also passed:
`nix build -L .#checks.aarch64-darwin.llzk-installcheck-release` rebuilt LLZK,
ran all 463 lit and 1,342 unit/C API tests, installed the package, and successfully
built/linked its CMake consumer. Package output:
`/nix/store/zdc7klicnyg2s1yj43fhpfabv0bbzf5k-llzk-release-3.0.0`.

## Binary R1CS conversion timing follow-up

The [29-circuit R1CS timing report](r1cs-conversion-timings.md) measures the
direct-field implementation through binary R1CS output, using three serial runs
per circuit. It separates frontend, evaluation/lowering, and binary export, and
includes all samples and observed timing ranges. These endpoints differ from the
evaluator-only measurements above.
