# Rolled Witness and Symbolic Constraint Evaluation Plan

## Objective

Replace the current requirement to inline components, unroll loops, and scalarize the entire LLZK
program before R1CS lowering with a split compilation model:

- Keep witness generation component-structured and rolled.
- Monomorphize definitions without inlining their bodies.
- Elaborate a concrete circuit instance graph and stable signal access paths.
- Symbolically execute constraint generation into one straight-line constraint representation.
- Run global constraint optimization and degree lowering on that representation.
- Assign physical R1CS wire numbers only after constraint optimization is complete.

The redesign should bypass the operations responsible for the current Poseidon flattening,
while-to-for, POD-to-scalar, and late scalarization costs. Existing performance branches are useful
as baselines, but the implementation should be developed as staged changes from `main` rather than
by extending the R1CS performance stack.

## Resulting compilation model

The source module continues to own the hierarchical storage and rolled witness program:

```text
Main::compute
  component calls remain calls
  loops remain loops
  runtime branches remain branches
  writes target the original component hierarchy
```

Constraint evaluation produces a separate canonical constraint program that references the same
storage through stable logical signal identities:

```text
Main::__llzk_flat_constrain
  one block
  no component constrain calls
  no loops
  no branches
  no aggregate construction or mutation
  scalar signal references, field expressions, guards, and constraint emissions only
```

The first implementation may represent the canonical program as a generated function on `Main`.
A dedicated flat-constraint operation or dialect can replace it once the required semantics are
proven. Keeping the existing structs avoids creating a duplicate scalar struct and a second source
of truth for signal storage.

## Compiler identities

Three identities must remain distinct.

### Specialization identity

A specialization denotes one concrete definition:

```text
(original symbol, concrete parameter tuple, concrete type arguments) -> SpecializationId
```

Definitions are cloned once per unique key. One specialization can serve many circuit instances.

### Instance identity

An instance denotes one occurrence of a specialized component in the elaborated circuit:

```text
(parent InstanceId, member symbol, aggregate indices) -> InstanceId
```

For example, every Poseidon round may share one specialization while retaining a different
`InstanceId`.

### Signal identity

A signal identity denotes one scalar witness location before backend layout:

```text
SignalKey = (InstanceId, member symbol, aggregate indices)
```

Human-readable paths such as `Main.rounds[7].sboxes[2].output` are diagnostics, not primary keys.
The compiler should intern symbols and index tuples rather than repeatedly constructing strings.

`SignalKey` is stable across constraint evaluation and optimization. `WireId` is a later,
backend-specific location. R1CS lowering creates the final mapping:

```text
SignalKey -> R1CS WireId
```

Other backends may instead map the same logical identity to an advice cell or trace column.

## Heterogeneous component arrays

An array whose element type depends on its index is a component family rather than an ordinary
homogeneous array. Elaboration records a descriptor:

```text
ComponentFamilyDescriptor {
  shape
  elementSpecialization(index tuple) -> SpecializationId
  elementInstance(index tuple) -> InstanceId
}
```

For `children[i] : Child<2*i + 1>` in `Parent<3>`, the descriptor resolves:

```text
children[0] -> instance of Child<1>
children[1] -> instance of Child<3>
children[2] -> instance of Child<5>
```

The source array and loops remain rolled. During static loop interpretation, the evaluator binds
the induction variable to a concrete integer for each iteration. An array read then uses the
descriptor to return a precise `(InstanceId, SpecializationId)` pair.

The first implementation requires an index selecting from a heterogeneous component family to be
known during constraint evaluation. Witness-dependent heterogeneous selection should produce a
clear unsupported diagnostic. A later extension may enumerate candidates under mutually exclusive
guards.

## Symbolic evaluator

The evaluator maintains an environment from MLIR SSA values to an abstract domain containing at
least:

```text
KnownIndex
KnownBool
ComponentRef(InstanceId, SpecializationId)
ComponentArrayRef(ComponentFamilyDescriptor)
SignalRef(SignalKey)
SignalArrayRef(base path, concrete shape)
PolynomialValue(shared expression DAG node)
```

Operation semantics are defined over these values:

- A member read from a `ComponentRef` consults the concrete specialization schema.
- A scalar signal member produces a `SignalRef`.
- A component member produces the child `ComponentRef` from the instance graph.
- A component-array member produces a `ComponentArrayRef`.
- A static array read refines the family descriptor with concrete indices.
- A constrain call pushes a new environment and interprets the cached specialized body.
- A static loop repeatedly interprets its body and maps yielded values to the next iteration.
- Field arithmetic builds hash-consed expression DAG nodes.
- Constraint operations append guarded equations to the flat constraint program.

Constraint calls are therefore semantically executed without cloning their operations into the
source IR. Static loops enumerate final constraints without first materializing unrolled LLZK.

### Conditionals

The evaluator carries a current guard, initially one. For a condition `b`:

```text
then guard = current guard * b
else guard = current guard * (1 - b)
```

Both branches are interpreted and emitted constraints retain an explicit guard until backend
lowering. Branch-produced scalar values merge through symbolic selects. Nested guards are shared
expression DAG nodes rather than duplicated trees.

The initial implementation permits dynamic conditions to select scalar values and constraints but
rejects dynamic selection between component identities or layouts. Boolean-producing operations
must establish the language's Boolean invariant; the conditional evaluator should not add a
duplicate Booleanity constraint for every use.

### Loops

The initial evaluator supports loops whose lower bound, upper bound, and step evaluate to known
integers. Loop-carried values are abstract values obtained from `scf.yield`. Unknown trip counts
that emit constraints are initially unsupported. A later bounded-loop extension may enumerate a
maximum range and guard iteration `i` with `i < runtimeBound`.

## Degree lowering and witness auxiliaries

Degree lowering runs on the flat constraint representation. A surviving auxiliary receives a new
logical identity and a defining witness expression:

```text
AuxSignalKey
constraint definition: aux = expression
witness definition:    aux := expression
```

For the hierarchical-storage implementation, auxiliaries can be private compiler-generated signal
members of `Main`. The rolled `Main::compute` receives an epilogue before every return that writes
the surviving auxiliary values. Packing, CSE, and dead auxiliary elimination must update the
constraint definition, member, witness assignment, and logical identity atomically.

Generated witness code may read internal component members after source access validation. Inner
`llzk.pub` attributes must not promote a leaf to a public circuit output when an outer access-path
member is private. Struct inlining and scalarization must derive public visibility from the entire
path instead of copying the innermost attribute.

## Staged implementation

Each stage should be developed on a fresh `codex/` branch from the latest `main` after the previous
stage lands. Do not stack these changes on the historical R1CS performance branches. If a required
correctness fix has not reached `main`, port that fix separately and minimally; do not import an
optimization stack merely to make a benchmark run.

### Stage 0: Semantics, observability, and benchmark harness

Suggested branch: `codex/symbolic-constraint-plan`.

Deliverables:

- Check in this design and implementation plan.
- Define counters and timing scopes for specializations, instances, signal paths, interpreted loop
  iterations, emitted constraints, expression DAG nodes, and peak resident memory.
- Add a benchmark manifest with named correctness and scale tiers.
- Add or adapt a driver that accepts a Circom corpus root, tool paths, benchmark filters, timeout,
  output directory, and a pipeline phase. It should write CSV/JSON summaries and per-stage logs.
- Disable intermediate IR dumps by default. Large dumps require an explicit opt-in flag.

Exit criteria:

- A Release build from `main` can run the selected existing pipeline and record a reproducible
  baseline for the benchmarks it currently supports.
- Poseidon inputs can be selected independently without copying generated multi-gigabyte IR into
  the repository.

### Stage 1: Definition monomorphization

Suggested branch: `codex/definition-monomorphization`.

Deliverables:

- Implement a worklist-driven specialization registry rooted at the concrete main component.
- Clone each struct or function definition once per unique concrete specialization key.
- Rewrite specialization-local symbol and type references without inlining calls or unrolling
  loops.
- Discover concrete specializations required by affine index-dependent component families.
- Detect non-terminating specialization recursion and report the specialization chain.
- Preserve source locations and record the original symbol on generated specializations.

Focused tests:

- Repeated uses of one parameter tuple share a specialization.
- Distinct tuples produce distinct definitions.
- Parameterized free and struct functions resolve correctly.
- Nested templates and affine-map parameters specialize correctly.
- No `scf.for`, `scf.while`, or function body is expanded.

Benchmark checkpoint:

- Run the monomorphizer only on the small Circom tier, Poseidon3, and Poseidon6.
- Record specialization count, pass time, peak RSS, and output operation count.
- Confirm operation growth follows unique specializations rather than component instance count.

### Stage 2: Instance graph, component families, and signal paths

Suggested branch: `codex/circuit-instance-elaboration`.

Deliverables:

- Build the concrete instance graph from `Main` without cloning specialized bodies per instance.
- Implement `InstanceId`, `SignalKey`, and canonical access-path interning.
- Implement component-family descriptors for index-dependent arrays, including nested and
  multidimensional affine maps.
- Derive public visibility from complete access paths.
- Add a printer/debug option for the instance graph and signal paths that is off by default.

Focused tests:

- `children[i] : Child<2*i+1>` resolves to the expected specializations and instances.
- Nested heterogeneous arrays resolve paths recursively.
- Repeated specializations have distinct instances and signal identities.
- A private top-level member keeps all flattened descendants private.
- A fully public path remains public.

Benchmark checkpoint:

- Run elaboration on the small tier, Poseidon3, and Poseidon6.
- Record instance count, signal-leaf count, descriptor lookup count, time, and peak RSS.
- Compare signal-leaf counts with existing R1CS wire accounting where the old pipeline succeeds.

### Stage 3: Core symbolic constraint evaluator

Suggested branch: `codex/symbolic-constraint-evaluator-core`.

Deliverables:

- Implement abstract environments and values for known indices, component references, signal
  references, and shared polynomial expressions.
- Interpret constants, field arithmetic, member reads, static array reads, constrain calls,
  straight-line returns, and `constrain.eq`.
- Interpret statically bounded `scf.for`, including loop-carried scalar values.
- Resolve heterogeneous component-array accesses from concrete evaluator indices.
- Emit a generated one-block constraint function alongside the original constrain function.
- Preserve provenance: original operation, specialization, instance path, call stack, and loop
  indices.
- Diagnose unsupported operations rather than falling back to IR inlining.

Focused tests:

- A call with polynomial arguments substitutes the caller expression correctly.
- A static loop emits the expected number of constraints without modifying the source loop.
- A heterogeneous component loop chooses the correct specialization on every iteration.
- Nested calls and nested arrays produce the expected signal paths.
- Shared expressions remain shared DAG nodes.

Benchmark checkpoint:

- Start with `test/FrontendLang/Circom/subcomponents_using_pod.llzk`,
  `circom_example_2B.llzk`, and `poseidon_m_concrete.llzk`.
- Run the external small tier, then Poseidon3 and Poseidon6.
- Compare canonicalized equations against the existing flattening path when it succeeds.
- Record emitted constraints, DAG nodes, wall time, peak RSS, and retained source operation count.

### Stage 4: Conditionals and remaining static control flow

Suggested branch: `codex/symbolic-constraint-control-flow`.

Deliverables:

- Interpret known `scf.if` by selecting one branch.
- Interpret dynamic scalar `scf.if` by propagating complementary guards.
- Merge scalar branch results with symbolic selects.
- Accumulate and share nested guards.
- Reject dynamic component identity, component layout, and heterogeneous-index selection with
  precise diagnostics.
- Add bounded handling for supported `scf.while` forms only if required by the benchmark tier;
  otherwise retain an explicit unsupported diagnostic.

Focused tests:

- Known true/false conditions emit only the selected constraints.
- Dynamic if/else constraints receive `b` and `1-b` guards.
- Nested conditions accumulate guards correctly.
- Branch-produced scalar values merge correctly.
- Type- or topology-changing dynamic branches fail intentionally.

Benchmark checkpoint:

- Add `early_return_loop_1.llzk`, `poly_expr_cleanup_1.llzk`, and
  `poly_expr_cleanup_2.llzk` to the recurring set.
- Rerun the small tier, Poseidon3, and Poseidon6.
- Record guarded constraint counts and maximum guard-DAG depth.

### Stage 5: Canonical flat constraint representation and global optimization

Suggested branch: `codex/flat-constraint-ir`.

Deliverables:

- Stabilize a canonical one-block constraint representation, either as a restricted generated
  function or dedicated operations.
- Replace repeated member-read chains with interned signal references or cached access-path reads.
- Add bounded canonicalization, constant folding, expression CSE, duplicate-constraint removal,
  and dead expression elimination.
- Preserve source and instance provenance through rewrites.
- Define deterministic printing for tests without requiring physical wire numbers.

Focused tests:

- Equivalent access paths intern to one signal reference.
- Repeated expressions share nodes.
- Duplicate constraints are removed without changing public signals.
- Large repeated patterns do not trigger quadratic rewrite behavior.

Benchmark checkpoint:

- Measure optimizer time independently on saved flat-constraint inputs.
- Track pre/post DAG nodes, constraints, and memory.
- Require no regression in Poseidon3 or Poseidon6 canonical constraint counts.

### Stage 6: Global degree lowering and rolled witness extension

Suggested branch: `codex/flat-degree-lowering`.

Deliverables:

- Run degree analysis on composed caller/callee expressions in the flat constraint program.
- Introduce auxiliary logical signals only at selected cut points.
- Generate defining constraints and compute-side assignments together.
- Add private compiler-generated auxiliary signal members to `Main` or an equivalent internal
  storage table.
- Insert dependency-ordered assignments before every rolled `Main::compute` return.
- Implement packing/rematerialization that removes unnecessary auxiliaries from both constraint
  and witness sides.
- Keep generated access privileges separate from public circuit visibility.

Focused tests:

- `(a+b)^2` does not require a boundary auxiliary.
- `(a*b)^2` receives the necessary degree-two factorization.
- Shared nonlinear expressions reuse an auxiliary when profitable.
- Packing removes a signal, defining constraint, member, and witness assignment atomically.
- Auxiliary definitions inside guarded constraints receive valid witness behavior on inactive
  branches.

Benchmark checkpoint:

- Compare auxiliary and final constraint counts against the existing poly-lowering pipeline.
- Verify generated witnesses satisfy the lowered constraints on the small tier.
- Rerun Poseidon3 and Poseidon6 and record auxiliary count, degree-lowering time, and peak RSS.

### Stage 7: R1CS lowering and final wire allocation

Suggested branch: `codex/flat-constraint-r1cs`.

Deliverables:

- Collect surviving logical signals after global optimization.
- Assign R1CS wires in the required public-output, public-input, private-input, and internal order.
- Emit and retain a deterministic `SignalKey -> WireId` map.
- Lower guarded constraints without exceeding the configured degree. Use shared guard auxiliaries
  rather than independently expanding every guarded equation.
- Teach witness serialization to use the final wire map, including degree-lowering auxiliaries.
- Compare canonical R1CS equations independently of incidental wire renumbering.

Focused tests:

- Public ordering follows complete access-path visibility.
- Nested private signals and compiler auxiliaries remain private.
- Wire aliases and eliminated internal signals are handled correctly.
- WTNS length and R1CS wire count agree.
- Existing and new lowering produce equivalent canonical equations on supported small circuits.

Benchmark checkpoint:

- Run the small external tier end to end against native Circom `--O0`.
- Run Poseidon3 and Poseidon6 end to end.
- Compare constraints, wires, nonzero terms, public counts, witness satisfaction, time, and peak
  memory.

### Stage 8: Corpus rollout and legacy-pipeline retirement decision

Suggested branch: `codex/symbolic-constraint-rollout`.

Deliverables:

- Add an opt-in pipeline flag first, then select the new path by default only after coverage and
  correctness criteria are met.
- Run benchmark clusters of 5-10 programs grouped by unsupported operation or failure class.
- Reduce every new compiler failure to a focused tracked LLZK test before changing implementation.
- Run the full Circom corpus at milestone boundaries.
- Identify old flattening/scalarization stages that remain necessary for other backends before
  removing or bypassing them.

Exit criteria:

- The selected Circom corpus emits valid R1CS or has a classified, tested unsupported feature.
- Small-circuit equations and witness behavior match the existing path or an explained correction.
- Native Circom size comparisons are recorded.
- Poseidon3 and Poseidon6 show the expected compile-time and memory improvement.
- PoseidonEx completes end to end within the agreed resource budget.

## Benchmark tiers

### Tier A: Focused tracked LLZK tests

Use small lit fixtures for every evaluator operation and failure mode. Initial existing inputs:

- `test/FrontendLang/Circom/subcomponents_using_pod.llzk`
- `test/FrontendLang/Circom/circom_example_2B.llzk`
- `test/FrontendLang/Circom/poseidon_m_concrete.llzk`
- `test/FrontendLang/Circom/early_return_loop_1.llzk`
- `test/FrontendLang/Circom/poly_expr_cleanup_1.llzk`
- `test/FrontendLang/Circom/poly_expr_cleanup_2.llzk`

Add reduced fixtures for heterogeneous component families, nested arrays, guards, and auxiliary
witness generation. CI must not depend on the external benchmark checkout.

### Tier B: Small external Circom smoke set

Use frontend-emitted LLZK from `/Users/shankarapailoor/veridise/circom-benchmarks`. The manifest
should include small representatives such as gates, `IsZero`, `Num2Bits`, `BinSum`, `Switcher`, and
the Poseidon `Sigma` component. This tier runs at every stage that can consume those programs.

### Tier C: Recurring scale set

- `poseidon3_test.circom` (`Poseidon(2)`)
- `poseidon6_test.circom` (`Poseidon(5)`)

These run after every stage's focused tests pass. Use Release `llzk-opt`, `--jobs 1`, and no
intermediate IR dumps. Record pass-local and total measurements separately.

Historical native Circom `--O0` constraint counts are 765 for Poseidon3 and 1,347 for Poseidon6;
regenerate and record the authoritative baseline when the new harness is established from `main`.

### Tier D: Broad functional clusters

Use prior benchmark classifications to select 5-10 circuits with shared features: basic gates,
arrays, cryptographic components, machine-learning circuits, nested component arrays, and dynamic
conditionals. Successful circuits remain in subsequent runs of their cluster.

### Tier E: PoseidonEx and full corpus milestones

- `poseidonex_test.circom` (`PoseidonEx(16, 17)`)
- The full Circom benchmark corpus

PoseidonEx should run only after a milestone passes Tiers A-C. Do not save textual intermediate IR.
Capture timing, peak memory, counters, final R1CS metadata, and concise logs, then delete transient
files. Do not repeatedly run PoseidonEx while developing a focused evaluator operation.

## Measurements and correctness checks

Every benchmark summary should include, when applicable:

- Git commit and pipeline configuration.
- Tool build type and frontend command.
- Stage status and first diagnostic.
- Wall time and peak resident memory by major pass.
- Input and output operation counts.
- Specialization, instance, and scalar signal counts.
- Interpreted loop iterations and constraint calls.
- Constraint count before and after optimization.
- Polynomial DAG nodes and maximum degree.
- Degree-lowering auxiliary count.
- Final R1CS wires, constraints, nonzero terms, and public counts.
- Witness generation success and constraint satisfaction.

Correctness comparison should not rely on textual operation order or incidental wire numbering.
For small circuits, canonicalize equations by logical signal path and normalized coefficients. For
large circuits, compare metadata first and use deterministic hashes of canonicalized constraints
when feasible.

## Branch and commit policy

- Create each implementation branch from the latest `main` after prerequisites merge.
- Keep correctness, infrastructure, and performance changes in separate commits.
- Do not cherry-pick the historical flattening, POD, while-to-for, or R1CS performance stacks into
  the redesign unless a specific correctness dependency is demonstrated.
- Keep generated benchmark output and large IR dumps out of Git.
- Rebuild `llzk-opt` after C++, TableGen, or build changes before manual benchmark runs.
- Run focused tests during development, `check-lit` before committing a stage, and `nix build -L`
  before declaring a build-affecting stage complete.

## Initial implementation order

The first coding branch should implement the specialization registry and worklist, but define the
identity types needed by Stage 2 at the same time so generated specializations do not acquire an
unstable naming contract. Once one heterogeneous component-array fixture resolves the correct
specialization for every static loop iteration, implement the minimal constraint evaluator and
compare its equations with the existing pipeline before expanding operation coverage.
