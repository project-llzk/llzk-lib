# PCL Backend {#pcl-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

Allows LLZK to be lowered to PCL (Picus Constraint Language) for use with the [Picus](https://docs.audithub.dev/picus-v2/) verifier

\include{doc} build/doc/mlir/dialect/PCLDialect.md

## Lowering pipeline

Run `llzk-opt input.llzk --llzk-full-pcl-lowering` to normalize while loops,
canonicalize loop bounds, flatten templates and unroll loops rooted at `llzk.main`,
then scalarize PODs and arrays before converting to PCL. Canonicalization also
runs after flattening and scalarization. Common subexpression elimination runs
before and after POD/array scalarization, followed by redundant read/write
elimination before PCL conversion.

This pipeline preserves component boundaries; it does not inline structs or free
functions. Existing PCL restrictions still apply, including the requirement that
each struct has a public output. In particular, assertion-only components and a
main component with no public outputs still require separate handling.

Use `--llzk-to-pcl` directly when the input has already been prepared.

When PCL is enabled, C clients can append the pipeline to a module pass manager
with `llzkBuildFullPCLLoweringPipeline`. Alternatively, call
`llzkRegisterPCLConversionPassPipelines` and parse
`builtin.module(llzk-full-pcl-lowering)` with `mlirParsePassPipeline`.

## Integer values

PCL felt constants and prime moduli use signed arbitrary-precision integers.
Arithmetic intermediates grow as needed and are reduced modulo the module's
prime. Literal storage widths have no semantic meaning. During LLZK-to-PCL
conversion, the prime modulus is inferred from the felt types used across the
circuit and stored in the top-level module's `pcl.prime` attribute. Conversion
requires a single field for the entire circuit. Its value must be at least two;
the PCL verifier does not check primality.
