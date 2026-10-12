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

## Child storage visibility preparation

`llzk-expose-child-members` makes subcomponent fields public, allowing the constraint
body to access felt leaves when flattening constraints without scalarizing struct
members.

The root's own member visibility is preserved because changing it would change
the circuit's public interface and hence its semantics.
