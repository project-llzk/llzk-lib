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
