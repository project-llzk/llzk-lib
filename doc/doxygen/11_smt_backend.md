# SMT Backend {#smt-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

LLZK uses the [upstream SMT dialect](https://mlir.llvm.org/docs/Dialects/SMT/) for solver
operations and terms. On top of that, LLZK defines the smt extensions below.

\include{doc} build/doc/mlir/dialect/SMTInfoDialect.md

Felt literals are reduced modulo the field selected by the lowering pass before
being emitted as SMT integer constants. This also applies to signed literals and
to source felt types whose field name is unspecified.
