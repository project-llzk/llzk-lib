# PCL Backend {#pcl-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

Allows LLZK to be lowered to PCL (Picus Constraint Language) for use with the [Picus](https://docs.audithub.dev/picus-v2/) verifier

\include{doc} build/doc/mlir/dialect/PCLDialect.md

## Integer values

PCL felt constants and prime moduli use signed arbitrary-precision integers.
Arithmetic intermediates grow as needed and are reduced modulo the module's
prime. Literal storage widths have no semantic meaning. During LLZK-to-PCL
conversion, the prime modulus is inferred from the felt types used across the
circuit and stored in the top-level module's `pcl.prime` attribute. Conversion
requires a single field for the entire circuit. Its value must be at least two;
the PCL verifier does not check primality.
