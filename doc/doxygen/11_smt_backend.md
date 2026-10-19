# SMT Backend {#smt-backend}

\htmlonly

<meta name="toc-level" content="2">
\endhtmlonly

\tableofcontents

LLZK uses the [upstream SMT dialect](https://mlir.llvm.org/docs/Dialects/SMT/) for solver
operations and terms. On top of that, LLZK defines the smt extensions below.

\include{doc} build/doc/mlir/dialect/SMTInfoDialect.md

Felt literals are reduced to `[0, p)` before being emitted as SMT integer
constants, including signed literals. A specified source field determines `p`;
for fieldless literals, the field selected by the lowering pass determines `p`.
