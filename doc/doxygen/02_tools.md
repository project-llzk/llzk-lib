# Tool Guides {#tools}

\htmlonly

<meta name="toc-level" content="3">
\endhtmlonly

\tableofcontents

# llzk-opt {#llzk-opt}

`llzk-opt` is a version of the [`mlir-opt` tool](https://mlir.llvm.org/docs/Tutorials/MlirOpt/) that supports
passes on LLZK IR files. You can refer to the `mlir-opt` documentation for a general
overview of the operation of `*-opt` tooling, but note that many options and passes
available in `mlir-opt` are not available in `llzk-opt`.
`llzk-opt -h` will show a list of all available flags and options.

#### LLZK-Specific Options

```
-I <directory> : Directory of include files
```

## LLZK Pass Documentation {#passes}

### Analysis Passes

\include{doc,raise=1} build/doc/mlir/passes/AnalysisPasses.md

### General Transformation Passes

\include{doc,raise=1} build/doc/mlir/passes/LLZKTransformationPasses.md

### 'array' Dialect Transformation Passes

\include{doc,raise=1} build/doc/mlir/passes/array/TransformationPasses.md

### 'polymorphic' Dialect Transformation Passes

\include{doc,raise=1} build/doc/mlir/passes/polymorphic/TransformationPasses.md

### Validation Passes

\include{doc,raise=1} build/doc/mlir/passes/LLZKValidationPasses.md

# llzk-lsp-server

`cmake --build <build dir> --target llzk-lsp-server` will produce an LLZK-specific
LSP server that can be used in an IDE to provide language information for LLZK.
Refer to the [MLIR LSP documentation](https://mlir.llvm.org/docs/Tools/MLIRLSP/) for
a more detailed explanation of the MLIR LSP tools and how to set them up in your IDE.

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref setup | \ref contribution-guide |
</div>
