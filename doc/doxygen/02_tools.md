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

# llzk-witgen {#llzk-witgen}

`llzk-witgen` executes LLZK witness-generation logic for the concrete main
component declared by `llzk.main`. It evaluates `compute()` and prints JSON
for either the public outputs of the main component or the full generated
witness signal set.

#### Basic Usage

```sh
llzk-witgen <input.llzk> --inputs <input.json>
```

#### LLZK-Specific Options

```
--inputs <file>              JSON file containing main compute inputs
-I <directory>               Directory of include files
--backend=<name>             Execution backend: interpreter or execution-engine
--output-scope=<name>        Output scope: public or full-witness
--dump-jit-core              Print the pre-LLVM JIT module
--dump-jit-llvm              Print the post-LLVM JIT module
```

#### Input Format

The `--inputs` file must contain a top-level JSON object or JSON array.

- A JSON object is keyed by `function.arg_name` attributes on the main
  `compute()` function arguments.
- A JSON array is interpreted positionally in declared argument order.

At the main boundary, `llzk-witgen` only supports `felt` and
`array<... x felt>` inputs, due to the restrictions posed on `llzk.main` components.
Field element values are accepted in the same JSON
form used by the witgen tests, namely JSON integers or decimal strings.

#### Output Format

`llzk-witgen` writes one JSON object to stdout. The exact shape depends on
`--output-scope`.

- `--output-scope=public` is the default.
  - The output JSON contains only the public outputs of the main component.
  - Public struct members become JSON object fields.
  - Public felt arrays become JSON arrays.
  - Field element leaves are rendered as decimal strings.
- `--output-scope=full-witness`
  - The output JSON contains two top-level objects: `inputs` and `signals`.
  - `inputs` records the main `compute()` arguments using their
    `function.arg_name` attributes when available, or stable fallback names
    such as `arg0`, `arg1`, and so on for positional inputs.
  - `signals` records all witness signals reachable from the returned main
    struct, including both public and private signals.
  - Non-signal leaves are omitted, though non-signal struct containers may
    still appear when needed to reach nested signals.
  - Felt arrays remain JSON arrays and field element leaves remain decimal
    strings.

#### Backends

`llzk-witgen` currently supports two execution backends:

- `--backend=interpreter`
  Executes LLZK `@compute` logic directly over the preprocessed MLIR.
- `--backend=execution-engine`
  Lowers preprocessed LLZK `@compute` IR to built-in MLIR dialects that can be
  natively converted to LLVM IR, then executes it with `mlir::ExecutionEngine`.

The default backend is `interpreter`, as the `execution-engine` does not currently
support all LLZK features due to existing lowering limitations (e.g., in the `-llzk-flattening` pass).

#### Preprocessing

Before execution, `llzk-witgen` performs the preprocessing needed to make
witness generation concrete and executable:

- `include` inlining
- flattening rooted at `llzk.main` when required
- affine lowering for execution-engine mode
- subcomponent inlining for execution-engine mode

Template parameters and affine instantiations are therefore supported only when
they can be fully resolved by the preprocessing pipeline before execution.

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
