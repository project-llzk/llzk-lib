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
Both emit passes reuse this normalization when needed.

`llzk-r1cs-lowering` accepts legacy flattened input and replaces structs with
circuits. It rejects modules marked `poly.evaluated_main`.

`llzk-r1cs-direct-lowering` requires `poly.evaluated_main` and the original storage
interface. It operates on main only, preserves structs and compute functions,
and emits `poly.wire_bindings` and `r1cs.main`. Do not pre-flatten evaluated input.

The registered pipelines are flat, explicit sequences:

- `llzk-full-r1cs-lowering`: legacy polynomial lowering, R1CS preparation,
  legacy R1CS emission, CSE.
- `llzk-full-direct-r1cs-lowering`: degree lowering, R1CS preparation,
  direct R1CS emission, CSE.

C++ callers select `R1CSLoweringMode::Legacy` or `R1CSLoweringMode::Direct` when
calling `buildFullR1CSLoweringPipeline`. The pipeline builder does not inspect IR.
