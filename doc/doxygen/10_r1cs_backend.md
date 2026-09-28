# R1CS Backend {#r1cs-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

\include{doc} build/doc/mlir/dialect/R1CSDialect.md

## Direct binary export

To evaluate constraints, lower their degree, and export binary R1CS in one process:

```sh
llzk-translate input.llzk --llzk-to-r1cs --r1cs-prime=<decimal-prime> -o circuit.r1cs
```

This runs monomorphization, symbolic constraint evaluation, and the full direct R1CS
lowering pipeline on the parsed module, then calls the binary exporter directly.
It does not print or reparse intermediate IR. Rolled compute and witness metadata
remain in memory during lowering. Already evaluated modules are also accepted.
The existing `--r1cs-to-binary` translation exports already lowered R1CS IR.

## Explicit lowering contracts

`llzk-r1cs-prepare` normalizes straightline constraints, creates auxiliary compute
assignments, and sets `r1cs.prepared`. It emits no circuits and preserves structs.
It prepares only main on evaluated modules, and all structs on legacy modules.
Both lowering modes reuse this normalization when needed.

`llzk-r1cs-lowering` defaults to direct lowering. It requires
`poly.evaluated_main` and the original storage interface. It operates on main
only, preserves structs and compute functions, and emits `poly.wire_bindings`
and `r1cs.main`. Run `llzk-monomorphize` and `llzk-evaluate-constraints` first;
do not pre-flatten evaluated input.

`llzk-r1cs-lowering=legacy=true` instead accepts legacy flattened input and
replaces structs with circuits. It rejects evaluated modules.

`llzk-full-r1cs-lowering` runs degree lowering, R1CS preparation, direct R1CS
emission, and CSE. Its input must already be evaluated as described above.
`llzk-full-r1cs-lowering=legacy=true` uses the legacy full polynomial lowering
pipeline and legacy R1CS emission instead.

C++ callers of `buildFullR1CSLoweringPipeline` default to direct lowering and
can explicitly select `R1CSLoweringMode::Legacy`. The builder does not inspect IR.
