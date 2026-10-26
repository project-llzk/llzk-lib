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

## Cast range checks

R1CS lowering rejects live `cast.tofelt` and `cast.toindex` operations and
potentially failing assertions in `constrain` functions. An assert
cast that can fail must be preserved even when its result is unused (current
lowering does not encode its range check). Valid constant assertions can be
folded by canonicalization before lowering (the full R1CS pipeline includes this).
Unused infallible casts, including boolean-to-felt casts and `cast.toindex`
assertions from fields whose entire canonical range fits in the index safe
range, such as BabyBear, are accepted even without prior cleanup.
