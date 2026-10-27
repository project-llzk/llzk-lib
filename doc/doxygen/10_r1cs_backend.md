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

## Lowering LLZK constraints

Run constraint flattening and degree lowering to degree two before
`-llzk-r1cs-lowering`. The pass checks the degree bound, adds auxiliary signal
fields and assignments in `@compute`, computes the signal layout, assigns R1CS
wire IDs, and converts the constraints. Each signal leaf receives a wire,
including unused signals. Private storage used in constraints needs `{signal}`.

R1CS IDs follow public outputs, public inputs, private inputs, then internal
signals, with wire zero reserved for constant one. The LLZK layout keeps its
own ordering. No LLZK path metadata is stored in the R1CS dialect.

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
