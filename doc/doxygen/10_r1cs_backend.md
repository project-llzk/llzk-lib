# R1CS Backend {#r1cs-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

\include{doc} build/doc/mlir/dialect/R1CSDialect.md

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
