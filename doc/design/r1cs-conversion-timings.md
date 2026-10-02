# R1CS conversion timings — 2026-09-27

All 29 benchmarks exported successfully in all three measured runs (87 exports).
These measurements use the direct nested-field constrain implementation and end at a binary R1CS file.

## Method

- Three sequential runs per benchmark; no additional benchmark jobs or builds were launched alongside this sweep.
- Median wall-clock seconds, with no cache flush or separate warm-up.
- Concrete Circom frontend mode, plaintext LLZK, debug information stripped.
- Frontend: source Circom to LLZK.
- Lowering: LLZK parsing/verification, monomorphization, symbolic constraint evaluation, degree lowering, R1CS lowering, and textual IR output.
- Export: parsing/verifying that textual IR and writing binary R1CS over BN254.
- Conversion: lowering + export within each trial, excluding the frontend.
- End-to-end: frontend + lowering + export within each trial; setup and cleanup are excluded.
- Each column is independently median-aggregated. A total can differ from the sum of the displayed stage medians.
- Three samples are descriptive, not a statistical speedup claim. Large SHA export timings varied substantially.
- Constraint and wire counts were consistent across repetitions. This timing sweep does not add witness-correctness coverage.

Platform: `macOS-14.4-arm64-arm-64bit-Mach-O`.

LLZK release binaries: `/nix/store/z59lbb7jagi0czpj0fspgxwnsky665kn-llzk-release-3.0.0/bin`.

Frontend: `/nix/store/6r86m4r79l33axagb1dq6b9m5caxlncv-circom-to-llzk-0.1.0/bin/circom`.

Manifest: `scripts/symbolic-evaluation-benchmarks.json`.

Commands for each trial:

```sh
circom "$source" --llzk concrete --llzk_plaintext --llzk_strip_debug_info -o "$tmp"
llzk-opt "$ir" --llzk-monomorphize --llzk-evaluate-constraints \
  --llzk-full-r1cs-lowering -o "$tmp/lowered.llzk"
llzk-translate "$tmp/lowered.llzk" --r1cs-to-binary \
  --r1cs-prime=21888242871839275222246405745257275088548364400416034343698204186575808495617 \
  -o "$tmp/circuit.r1cs"
```

## Results

| Benchmark | Constraints | Frontend s | Lowering s | Export s | Conversion s | Conversion range s | End-to-end s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| isequal | 4 | 0.030 | 0.014 | 0.012 | 0.026 | 0.024–0.669 | 0.056 |
| greaterthan | 39 | 0.023 | 0.015 | 0.014 | 0.028 | 0.028–0.030 | 0.052 |
| mux1_1 | 14 | 0.025 | 0.015 | 0.013 | 0.028 | 0.027–0.029 | 0.053 |
| mux2_1 | 28 | 0.026 | 0.017 | 0.014 | 0.031 | 0.031–0.031 | 0.057 |
| mux3_1 | 47 | 0.027 | 0.018 | 0.015 | 0.034 | 0.033–0.034 | 0.061 |
| babyadd_tester | 6 | 0.029 | 0.011 | 0.012 | 0.023 | 0.023–0.023 | 0.052 |
| babycheck_test | 3 | 0.029 | 0.011 | 0.011 | 0.022 | 0.022–0.025 | 0.051 |
| edwards2montgomery | 2 | 0.014 | 0.011 | 0.011 | 0.022 | 0.022–0.022 | 0.036 |
| montgomery2edwards | 2 | 0.014 | 0.011 | 0.011 | 0.022 | 0.021–0.022 | 0.036 |
| montgomeryadd | 3 | 0.015 | 0.011 | 0.012 | 0.023 | 0.023–0.024 | 0.038 |
| montgomerydouble | 4 | 0.015 | 0.011 | 0.011 | 0.023 | 0.022–0.024 | 0.037 |
| sum_test | 200 | 0.030 | 0.023 | 0.025 | 0.047 | 0.047–0.048 | 0.077 |
| sign_test | 521 | 0.045 | 0.157 | 0.074 | 0.231 | 0.230–0.239 | 0.276 |
| constants_test | 33 | 0.016 | 0.013 | 0.013 | 0.026 | 0.026–0.026 | 0.042 |
| pointbits_loopback | 5,687 | 0.161 | 0.998 | 0.629 | 1.627 | 1.612–1.656 | 1.785 |
| babypbk_test | 10,114 | 0.153 | 0.608 | 0.453 | 1.097 | 1.049–1.202 | 1.249 |
| escalarmul_min_test | 6,150 | 1.723 | 1.857 | 1.247 | 3.105 | 3.089–3.293 | 4.827 |
| escalarmulany_test | 8,131 | 0.129 | 0.410 | 0.329 | 0.752 | 0.719–0.796 | 0.881 |
| escalarmulfix_test | 10,114 | 0.154 | 0.571 | 0.435 | 1.006 | 0.954–1.042 | 1.163 |
| pedersen_test | 13,108 | 4.598 | 4.161 | 2.840 | 7.001 | 6.913–7.128 | 11.659 |
| pedersen2_test | 8,127 | 0.145 | 0.459 | 0.372 | 0.826 | 0.820–0.860 | 0.971 |
| sha256_test448 | 408,640 | 2.398 | 37.465 | 34.842 | 76.737 | 68.918–118.432 | 79.017 |
| sha256_test512 | 408,640 | 2.367 | 41.084 | 54.033 | 95.116 | 82.714–135.603 | 97.483 |
| sha256_2_test | 204,465 | 1.582 | 16.733 | 23.786 | 40.519 | 34.186–55.494 | 42.101 |
| eddsa_test | 45,259 | 0.531 | 4.377 | 3.063 | 7.395 | 7.028–8.138 | 7.925 |
| eddsamimc_test | 21,737 | 0.338 | 1.723 | 1.114 | 2.856 | 2.838–2.951 | 3.194 |
| eddsaposeidon_test | 21,246 | 1.375 | 2.931 | 1.809 | 4.740 | 4.455–5.284 | 6.115 |
| smtverifier10_test | 12,582 | 2.033 | 3.642 | 2.405 | 6.047 | 5.786–6.241 | 8.117 |
| smtprocessor10_test | 20,465 | 2.008 | 5.266 | 3.671 | 8.937 | 8.594–9.483 | 10.975 |

[CSV summary](r1cs-conversion-timings.csv) and [all samples and tool metadata](r1cs-conversion-timings.json).

## Profiling follow-up

A separate diagnostic run of `sha256_2_test` (204,465 constraints) used
`--mlir-timing --mlir-timing-display=list` and macOS `sample`. This is not an
additional median benchmark. The lowering command took 16.47 seconds wall time:

| Work | Reported wall seconds |
| --- | ---: |
| R1CS lowering | 4.282 |
| Textual output | 4.075 |
| CSE | 3.311 |
| Symbolic evaluation | 3.228 |
| Degree lowering | 0.617 |

The full intermediate text was 222.45 MiB, versus a 22.14 MiB binary. The evaluated
constrain function header alone occupied 36.06 MiB, and the circuit header
(including the wire map) occupied 26.45 MiB. Lines containing poly metadata
occupied 120.71 MiB; that last number also includes the operations on those lines
and is not a measurement of metadata bytes alone.

In a single unprofiled comparison, full-module binary export took 14.73 seconds;
exporting a temporary R1CS-only module took 7.99 seconds. The resulting binary
files were byte-identical. The R1CS-only text still occupied about 89 MiB. Peak
RSS was about 1.01 GiB for full-module export and 0.95 GiB for circuit-only export.
A separate sampled full-module export spent approximately two-thirds of sampled
main-thread stacks under source parsing/verification, prominently parsing the
retained function attributes. Sampling began one second after launch, so these
are diagnostic proportions rather than whole-process timing percentages.

This supports prioritizing an in-memory lowering-to-binary path, followed by
reducing duplicated textual binding/provenance data. Rolled compute and the
witness mapping should remain available for witness generation. The profiling
does not establish the cause of the wide export-time variation across trials.

## In-memory follow-up

The [implemented in-memory translation and SHA rerun](r1cs-inmemory-sha.md)
avoid printing and reparsing the intermediate module. The three SHA circuits
were remeasured over three fixed-input trials and checked against fresh two-tool
conversions byte for byte.
