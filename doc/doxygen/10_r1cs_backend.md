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

R1CS lowering rejects residual `cast.tofelt` operations and `cast.toindex`
operations with `assert` overflow semantics in `constrain` functions. An assert
cast must be preserved even when its result is unused, but the current lowering
does not encode its range check. Valid constant assertions can be folded by
canonicalization before lowering (the full R1CS pipeline includes this).
