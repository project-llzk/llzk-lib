# R1CS Backend {#r1cs-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

\include{doc} build/doc/mlir/dialect/R1CSDialect.md

## Direct binary export

To evaluate constraints, lower their degree, and export binary R1CS in one process:

```sh
llzk-translate input.llzk --llzk-to-r1cs -o circuit.r1cs
```

The modulus defaults to the unique field used by the module's LLZK felt types,
including built-in fields such as `bn128` and custom fields. Pass
`--r1cs-prime=<decimal-prime>` to override it or when the input does not specify
one unique field. Bare R1CS IR whose LLZK field types have been erased still
requires the explicit option. This applies to both binary export commands.

This runs monomorphization, symbolic constraint evaluation, and the full direct R1CS
lowering pipeline on the parsed module, then calls the binary exporter directly.
It does not print or reparse intermediate IR. Rolled compute and witness metadata
remain in memory during lowering. Already evaluated modules are also accepted.
The existing `--r1cs-to-binary` translation exports already lowered R1CS IR.

### LLZK layout map

Pass `--llzk-layout-map=circuit.llzk-layout` with either binary translation to
export an optional text sidecar. It is available for circuits produced by direct
R1CS lowering. The exporter writes this file after the binary R1CS stream; no
lowering pass performs filesystem output.

The deterministic version-1 format assigns logical signal ids by structural
path order, independently of physical wire order or MLIR's attribute printer.
Roots and array indices compare numerically; member names compare
lexicographically. Numeric segments precede string segments, and a path precedes
any longer path for which it is a prefix. Root zero (`main`) therefore precedes
argument roots, and array index 2 precedes index 10.
Argument SSA names do not supply names: use the explicit `function.arg_name`
argument attribute. The map lists signal ids and their access paths, then maps
physical R1CS wires to those ids. The R1CS section
always contains `wire 0<TAB><one>` for the implicit constant-one wire. The first
path component is `main` for the main circuit instance or `arg["<name>"]` for
a named constrain-function argument; unnamed arguments use `arg<N>`. Members
use MLIR string literals in brackets and array elements use decimal indices; for example,
`signal 4<TAB>main["children"][0]["out"]`.

The `wire` fields remain in the order used by binary R1CS and WTNS export. The
exporter verifies that the direct-lowering bindings match the complete physical
R1CS wire layout and each wire's attached logical signal. Requesting a layout
map for an R1CS circuit without those bindings is an error.

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
