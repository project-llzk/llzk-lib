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

# llzk-translate {#llzk-translate}

`llzk-translate` is a version of the `mlir-translate` tool. Includes translations for backends 
specific to LLZK along with translations available in upstream MLIR (i.e. LLVM or C++).
`llzk-translate -h` will show a list of all available flags and options.

The tool expects that the IR has already been converted to the backend's dialect IR. For example:

```sh 
llzk-opt <input.llzk> --llzk-to-pcl | llzk-translate --pcl-to-lisp
```

#### LLZK-Specific Options

```
--pcl-to-lisp           Translates from PCL IR to PCL lisp
--smt-to-smtlib         Translates from SMT to SMTLIB
--zklean-to-lean        Translates from zkLean dialects IR to Lean code
```

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
--uninitialized-behavior     Control default handling of uninitialized witness values
--uninitialized-seed         Seed used for randomized uninitialized witness values
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

# llzk-smt-check {#llzk-smt-check}

`llzk-smt-check` runs an external SMT solver on an SMT-LIB 2 script and reports
the result of each `check-sat` stage. It is intended to consume the staged
SMT-LIB emitted by `llzk-opt --smt-to-smtlib`, including explicit metadata
produced from `smt.set_info` ops such as `(set-info :llzk-stage "pre")` and
`(set-info :status unsat)`.

#### Basic Usage

```sh
llzk-smt-check input.smt2
llzk-opt --smt-to-smtlib -o /dev/null input.llzk | llzk-smt-check -
```

`--smt-to-smtlib` exports the unique top-level `smt.solver` directly contained
in the root `builtin.module`. Module-scope `func.func` definitions are treated
as helpers that may be called from within that solver body. Exported scripts
preserve any `smt.set_logic` operation in that solver and otherwise begin with
the conservative fallback `(set-logic ALL)`. Any staged metadata consumed by
`llzk-smt-check` must be introduced explicitly in the SMT dialect IR using
`smt.set_info`, for example:

```mlir
smt.set_info ":llzk-root" "CheckGate"
smt.set_info ":llzk-stage" "pre"
smt.set_info ":status" unsat
```

`smt.check` lowers only to a bare `(check-sat)`. Because SMT-LIB scripts cannot
branch on `check-sat` results internally, `--smt-to-smtlib` accepts only
`smt.check` operations whose `sat`, `unknown`, and `unsat` regions are empty.
Result-dependent script behavior would need a higher-level driver format outside
SMT-LIB.

#### LLZK-Specific Options

```
--solver-binary=<path>      SMT solver executable to run (default: z3)
--quiet                     Suppress per-stage summaries and rely on exit status
--dump-raw-output           Print raw solver stdout after the stage summaries
```

#### Behavior

- The tool reads SMT-LIB from a file or stdin.
- `set-info :status <sat|unsat|unknown>` provides the expected result for the
  subsequent `check-sat`.
- `set-info :llzk-stage "<name>"` and `set-info :llzk-root "<name>"` provide
  LLZK-specific stage and root labels for the subsequent summaries.
- When expected-result annotations are present, every reported solver result
  must match the expected `sat`, `unsat`, or `unknown` result for the tool to
  succeed.
- Without annotations, the tool still executes the script and labels checks as
  `check[0]`, `check[1]`, and so on.
- Any solver launch failure, malformed output, result-count mismatch, or stage
  mismatch causes a non-zero exit status.

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
