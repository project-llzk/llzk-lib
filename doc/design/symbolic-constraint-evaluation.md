# Rolled symbolic constraint evaluation

The opt-in pipeline is:

```sh
llzk-opt input.llzk --llzk-monomorphize --llzk-evaluate-constraints
```

The evaluator executes specialized constrain bodies directly. It does not run an
unroller, inliner, scalarizer, or legacy flattening pass. Original templates,
specialized definitions, compute functions, and rolled constrain functions remain
unchanged. A preservation regression compares the complete retained text before
and after evaluation.

The result is a module-level `function.def @__llzk_flat_constrain` with one block,
scalar arguments, scalar expressions, and `constrain.eq` operations. It has no
component calls, loops, branches, aggregate construction, or aggregate mutation.
A module-level function with explicit storage bindings avoids introducing private
member reads or changing source access-control rules. This is an intermediate
constraint program, not a replacement witness entrypoint or a completed R1CS
backend. The existing `llzk.main` and legacy pipelines are unchanged.

## Interpreter and identities

`SymbolicConstraintEvaluation.cpp` owns the implementation. Each invocation of a
source function gets an independent SSA environment. Abstract values contain
known constants, scalar expressions, lazy hierarchical witness references, or
aggregate contents. Array/POD reads, writes, and call argument passing preserve
value-copy semantics. Copying storage references preserves logical signal identity.

The interpreter supports static `scf.for` with loop-carried values, static
`scf.while`, known `scf.if`, and single-block `scf.execute_region`. Field constants
and MLIR folding resolve the field-valued counters emitted by Circom. Constant
globals, multidimensional array accesses, PODs, and specialized free-function calls
are interpreted. Mutable aggregate storage exists only in the interpreter.

Component occurrences are interned by structural access path as they are visited.
A component array's `poly.family` descriptor resolves the actual specialization
using the complete concrete index tuple. Calls dispatch on that instance's
specialization, rather than the unspecialized family signature. Many instances
can share one specialized definition. Instance counts describe visited occurrences;
this is lazy elaboration, not a complete enumeration of unused component storage.

The serialized contract is:

- `poly.instances`: occurrence ID, parent ID when applicable, specialization ID,
  and absolute storage path. IDs are local to this evaluation/module.
- `poly.signal_bindings`: one entry per flat function argument, containing owning
  instance ID, relative `member_path`, absolute `path`, and effective public flag.
- Paths use interned `StringAttr` symbols and integer indices. The first absolute
  path element is the original main constrain argument number (`0` is self).
- `ConstraintEvaluation.h` exposes `InstanceId`, `SignalKey`, and a binding reader.
  A signal key is `(InstanceId, member_path)`, never a physical wire number.
- `poly.source` links the flat function to the specialized source constrain method.
- Each equation retains its source location, source-operation traversal ordinal,
  call frames (callee and component instance), and local loop iteration/induction
  values. Ordinals refer to the retained module's postorder traversal before the
  generated function is appended. Existing specialization origin/argument metadata
  remains available on the referenced definitions.

Public visibility is the conjunction along the complete storage path. A public
child output beneath a private parent member remains private. Main input bindings
use the source argument's public attribute. Expression CSE and constant interning
share scalar DAG nodes without merging distinct signal locations.

## Boundaries

Dynamic conditions, dynamic loop trip counts, dynamic array selection, unresolved
array shapes, column offsets, unsupported effects, and uninitialized values used
as scalars are rejected. Static loops are bounded by `max-steps` (default ten
million interpreted operations); calls also have a depth limit. Failed evaluation
removes its partially generated function. Source IR is not rewritten as a fallback.
Re-running on a module already containing the generated symbol is diagnosed.

Two existing plan fixtures also expose remaining boundaries:
`test/FrontendLang/Circom/poseidon_m_concrete.llzk` fails before evaluation when
monomorphization gives incompatible 3-by-3 and 2-by-2 return branches the same
concrete return type. `early_return_loop_1.llzk` is rejected when its helper uses
an uninitialized Boolean value. They are not counted as passing evaluator cases.

Dynamic guards/select merging, degree lowering, auxiliary witness assignments,
full eager instance-graph enumeration, and physical wire layout remain later work.
Backend consumers must interpret the binding metadata; passing the generated
function to an old backend alone does not establish that integration. Nonconstant
scalar operations are retained with their original semantics; no claim is made
that every accepted scalar expression is already quadratic R1CS form.

## Regression and semantic coverage

Focused tests cover repeated instances of one definition, polynomial call
arguments, carried loop values, one- and two-dimensional heterogeneous families,
aggregate value copies, multidimensional globals, static while loops, shared
expressions, private/public paths, and rejected dynamic bounds. FileCheck patterns
are generated from the emitted function using `scripts/generate-test-checks.py`
after a successful temporary-output dry run.

The repeated-instance regression also compares normalized polynomial equations
with `--llzk-full-struct-inlining`, using logical signal paths instead of SSA names.
Both produce `children[0].out = x*x` and `children[1].out = x*x*x*x`.
The legacy output promotes private-path leaves to public; the new binding flags
correctly retain the private outer member, as required by the design.

The external small manifest has 15 successful templated frontend inputs. All 15
pass monomorphization, symbolic evaluation, and output verification. BinSum fails
in frontend `poly.expr` construction before either LLZK pass runs.

`scripts/check-symbolic-poseidon.py` checks the emitted scalar equation subset
against native Circom witnesses. From the main inputs it propagates directed
equations, requires every referenced signal to be determined, verifies every
equation, and compares the public output with native witness wire 1. It also
perturbs each solved signal by one and requires a failing equation. Three vectors
per circuit (zeros, consecutive positive integers, and multiples of 123456789)
passed all checks: 767 perturbations per Poseidon3 vector and 1,352 per Poseidon6
vector. An independent binary R1CS reader also checked all native equations
against all six native witnesses; all were satisfied. Both emitted functions
contain zero loops/calls and exactly one public binding. This is reproducible
semantic evidence, not an all-input equivalence proof
or a comparison of every native intermediate wire's identity.

## Poseidon3 and Poseidon6 measurements

Measured on macOS arm64, Release LLZK/LLVM/MLIR, serially with one benchmark job.
The frontend is Circom 2.2.2 at `bf3b28ea35f7530c6ce3e60c77532f5f9b9f69a7`.
The source checkpoint is `52a7466b554558088076735387c293db9426725c` plus the recovered
monomorphizer fixes and this evaluator. No PoseidonEx benchmark was run.

| Metric | Poseidon3 | Poseidon6 |
| --- | ---: | ---: |
| Frontend wall seconds | 1.606 | 1.244 |
| Frontend peak MiB | 151.19 | 164.36 |
| Input source operations | 18,343 | 19,042 |
| Specialized definitions | 75 | 78 |
| Specialization pass milliseconds | 10.46 | 10.09 |
| Retained operations after specialization | 36,539 | 37,934 |
| Visited component instances | 156 | 186 |
| Referenced scalar signals | 767 | 1,352 |
| Interpreted loop iterations | 1,575 | 3,405 |
| Interpreted calls, including free functions | 159 | 189 |
| Emitted equations | 765 | 1,347 |
| Shared nonconstant scalar expression nodes | 1,026 | 2,268 |
| Generated function operations | 2,115 | 4,386 |
| Evaluation pass milliseconds | 34.18 | 68.56 |
| Combined llzk-opt process wall seconds | 0.720 | 0.774 |
| Combined llzk-opt peak MiB | 63.78 | 69.97 |
| Native O0 constraints | 765 | 1,347 |
| Native O0 wires (including constant wire) | 768 | 1,353 |

Pass times exclude parsing and verification; process measurements include parsing,
both passes, and verification. Peak memory is process-wide, not a separately
attributable evaluator peak. These are single samples, not statistical speedup
claims. Fresh concrete frontend output uses compact global constant arrays; its
operation counts differ from the older checkpoint measurements, so those totals
are not an apples-to-apples evaluator comparison. No degree-lowering optimization
or duplicate-equation elimination is applied here.

The scale inputs use **concrete frontend mode**. Templated Poseidon translation
still fails with `variable 'nInputs' not found`. Investigation found that
`BlockGenContext` inherits the empty `poly_template_binding_names` default when
constructing nested `poly.expr` initializers. Forwarding these names alone was
already shown by the preceding checkpoint to expose recursive expression
construction and illegal initializer dependencies. No speculative frontend patch
is retained. The concrete runs establish loop/call interpretation and constraint
semantics, not successful templated Poseidon specialization. The family fixtures
and small templated corpus separately exercise actual specialization.

Example measurement command (temporary frontend IR is deleted by the driver):

```sh
nix develop .#release --command bash -c 'python3 scripts/benchmark-monomorphization.py \
  --corpus /Users/shankarapailoor/veridise/circom-benchmarks \
  --frontend /Users/shankarapailoor/veridise/circom/result/bin/circom \
  --llzk-opt build/bin/llzk-opt --phase evaluate --frontend-mode concrete \
  --plaintext --tier scale --jobs 1 --build-type Release \
  --output /tmp/llzk-eval-scale'
```

Native references were regenerated with Circom `--O0 --r1cs --wasm`. Temporary
LLZK, R1CS, WASM, witnesses, and raw logs are removed after summarizing results.

## Final build validation

`nix build -L` succeeded for the implementation, producing
`/nix/store/bwj2iplnm0f626h5b6f3lasz0xxc9jyb-llzk-release-3.0.0`.
The package passed 456 lit tests (four unsupported, one expected failure) and all
1,340 unit/C API tests. The standalone CMake `check-lit` target also passed.
No frontend changes were retained, and the originating worktree was not modified.
The recovered checkpoint, including its eight monomorphizer fixes and runnable
examples, remains included. Temporary benchmark artifacts have been removed.

## Expanded corpus sweep

See [29 additional benchmarks](symbolic-evaluation-expanded.md) for both frontend
modes, native O0 counts, scaling measurements, and the newly exposed failures.
