# R1CS Backend {#r1cs-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

\include{doc} build/doc/mlir/dialect/R1CSDialect.md

## Arbitrary-precision coefficients

R1CS constants and multipliers use exact signed mathematical integers. Their
assembly has no integer-width annotation: for example,
`r1cs.mul_const %x, -1 : !r1cs.linear`. Binary export reduces each coefficient
modulo the selected prime. The `--r1cs-prime` decimal option does not require a
bit width; the exporter derives the fixed byte width required by the R1CS format.

## Layout-based lowering

`-llzk-constraint-flattening-r1cs-lowering` runs constraint flattening, degree lowering for
`llzk.main`, R1CS conversion, and common subexpression elimination. It retains
aggregate storage rather than running storage scalarization. Source helper and
component definitions are not independently converted into extra circuits.

The R1CS conversion then:

1. Checks that the constraints are polynomial and have degree at most two.
2. Adds auxiliary signal fields and equations where needed to express each
   constraint as A * B = C, with assignments for those fields in `@compute`.
3. Computes the LLZK layout, including the auxiliary signals.
4. Assigns R1CS wire IDs: zero for constant one, then public outputs, public
   inputs, private inputs, and internal signals.
5. Creates the wires and converts the constraints to use them.

LLZK layout IDs and R1CS wire IDs may differ. Each CSV row identifies the same
signal by its access path and records both IDs. The pass writes these rows
directly from its layout and wire map. It does not store LLZK layout metadata in
R1CS IR. Assigning R1CS IDs does not change the LLZK layout.

Emit all three outputs in one lowering run: LLZK with auxiliary fields and
assignments in `@compute`, the layout CSV, and R1CS IR:

```sh
llzk-opt \
  --llzk-constraint-flattening-r1cs-lowering='output-llzk=circuit.aux.llzk output-layout=circuit.csv' \
  circuit.llzk -o circuit.r1cs.mlir
llzk-translate --r1cs-to-binary --r1cs-prime=2013265921 circuit.r1cs.mlir -o circuit.r1cs
```

The pass saves `circuit.aux.llzk` after adding all auxiliary fields, constraints,
and assignments in `@compute`, before replacing LLZK with R1CS. Users can use
that LLZK and the CSV to generate their own witnesses. The pass does not produce
`.wtns` files. `output-llzk` and `output-layout` also work directly on
`-llzk-r1cs-lowering` and on the legacy `-llzk-full-r1cs-lowering` pipeline.
The two filenames must be distinct and cannot be `-`.

CSV columns are `access_path,llzk_layout_id,r1cs_wire_id`. Rows follow LLZK
layout order and include auxiliaries and unused signals. Access paths use the
same spelling as `.llzk-layout`; quoting preserves commas and embedded quotes.
Constant one has no LLZK layout ID and is omitted. The CSV is emitted during
lowering, while the LLZK paths are available. The R1CS dialect and its binary
exporter do not interpret those paths.

The binary exporter assigns wire indices during serialization. For circuits
produced by layout-based lowering, these indices agree with the wire IDs
assigned during lowering, giving an identity wire-to-label table.

Private storage without `{signal}` is not implicitly promoted to an R1CS wire.

Constraint statements follow source equality order. Within each linear combination,
repeated reads of the same signal are combined and nonzero terms are emitted in
increasing LLZK layout-ID order. This makes operand and statement ordering independent
of SSA addresses and hash-map iteration.

## Cast range checks

R1CS lowering rejects live `cast.tofelt` and `cast.toindex` operations and
potentially failing assertions in `constrain` functions. An assert
cast that can fail must be preserved even when its result is unused (current
lowering does not encode its range check). Valid constant assertions can be
folded by canonicalization before lowering (the full R1CS pipeline includes this).
Unused infallible casts, including boolean-to-felt casts and `cast.toindex`
assertions from fields whose entire canonical range fits in the index safe
range, such as BabyBear, are accepted even without prior cleanup.
