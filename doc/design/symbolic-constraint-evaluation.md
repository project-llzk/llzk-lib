# Rolled symbolic constraint evaluation

The opt-in pipeline is:

```sh
llzk-opt input.llzk --llzk-monomorphize --llzk-evaluate-constraints
```

The evaluator executes specialized constrain bodies after cleanup on a temporary
module copy. Cleanup folds operations, removes dead computations, and simplifies
`scf.if`; it does not register loop-unrolling patterns. Unused calls are removed
only when a transitive callee scan proves absence of observable effects.
Constraints, assertions, mutations, unknown operations, unavailable bodies, and
recursive call cycles prevent this removal. Fresh `llzk.nondet` values are local
and removable when unused, but are never merged as constants. This deliberately
conservative proof does not attempt escape analysis of local mutable arrays.

The evaluator propagates both attribute-valued and SSA-valued folds through its
abstract environment, including selects whose conditions become known only in a
call frame or loop iteration. No SSA value owned by a temporary fold operation
is retained.

The evaluator executes cleaned specialized constrain bodies directly, without
unrolling or inlining the compute program. It builds a **new straight-line
`@constrain` method in the specialized main struct**, retaining the original
`(%self, inputs...)` signature. The previous specialized main constrain body is
replaced after successful evaluation. Source templates, other specialized methods,
and all rolled compute bodies are retained.

Witness operands are actual chains of `struct.readm`, constant-index `array.read`,
and POD reads rooted in those arguments. They are not extra scalar function
arguments. Nested struct members are marked `llzk.pub` to make these accesses
legal; `poly.original_public` records their circuit visibility before promotion.
Main member visibility need not change because the generated method belongs to
Main. Signal-binding metadata annotates the reads for wire allocation and
provenance, rather than standing in for executable reads.

The direct binary lowering path keeps the parsed module in memory throughout:

```sh
llzk-translate input.llzk --llzk-to-r1cs --r1cs-prime=<decimal-prime> -o circuit.r1cs
```

To inspect the intermediate IR, the equivalent two-tool path is:

```sh
llzk-opt input.llzk --llzk-monomorphize --llzk-evaluate-constraints \
  --llzk-full-r1cs-lowering -o lowered.llzk
llzk-translate lowered.llzk --r1cs-to-binary --r1cs-prime=<decimal-prime> -o circuit.r1cs
```

For witness generation, save the evaluated module before R1CS export and pass it
to `llzk-witgen ... --output-wtns witness.wtns`. The tool prepares degree and R1CS
auxiliary assignments before executing compute. `--llzk-poly-lowering-pass=max-degree=2` and
`--llzk-r1cs-prepare` expose these stages separately.

On evaluated modules, the full R1CS pipeline skips legacy flattening/inlining,
lowers only the generated main constrain method, and adds auxiliary assignments
to rolled main compute returns. Export leaves the LLZK program alongside the
R1CS circuit; the translator accepts both dialect sets. The circuit's
`poly.wire_bindings` records storage paths in physical wire order: public outputs,
public inputs, private inputs, and internal signals (wire zero is implicit one).
Public classification uses the original complete-path visibility. Circuit
attributes, including this mapping, survive textual IR roundtrips.

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
- `poly.signal_bindings`: one entry per referenced scalar storage location, containing owning
  instance ID, relative `member_path`, absolute `path`, and effective public flag.
- Paths use interned `StringAttr` symbols and integer indices. The first absolute
  path element is the original main constrain argument number (`0` is self).
- `ConstraintEvaluation.h` exposes `InstanceId`, `SignalKey`, and a binding reader.
  A signal key is `(InstanceId, member_path)`, never a physical wire number.
- `poly.source` links the generated method to the original source constrain method.
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

Observable dynamic conditions, dynamic loop trip counts, dynamic array selection, unresolved
array shapes, column offsets, unsupported effects, and uninitialized values used
as scalars are rejected. Static loops are bounded by `max-steps` (default ten
million interpreted operations); calls also have a depth limit. Failed evaluation
discards the temporary module and its partially generated function. Source IR is not rewritten as a fallback.
Re-running on a module already containing the generated symbol is diagnosed.

Two existing plan fixtures also expose remaining boundaries:
`test/FrontendLang/Circom/poseidon_m_concrete.llzk` fails before evaluation when
monomorphization gives incompatible 3-by-3 and 2-by-2 return branches the same
concrete return type. `early_return_loop_1.llzk` is rejected when its helper uses
an uninitialized Boolean value. They are not counted as passing evaluator cases.

Dynamic guards/select merging and full eager instance-graph enumeration remain
outside the evaluator. Degree and R1CS normalization are now integrated for the
supported polynomial subset. Evaluation alone does not guarantee that arbitrary
non-polynomial scalar operations are lowerable to R1CS. Witness tests cover
concrete nested structs/arrays and auxiliary assignments; family fixtures also
check verified direct-read IR but do not initialize a complete family witness.

## Direct-read integration validation

`direct-pipeline_pass.llzk` and `direct-array-pipeline_pass.llzk` check binary
R1CS/WTNS agreement for nested scalar and array storage, using one nonzero input
per fixture. Independent tests cover auxiliary input capture, degree tightening,
wire visibility, unused inputs, translation equivalence, and idempotence.
`witness_constraint_violation_fail.llzk` checks rejection of a corrupted output
witness separately from successful witness validation. Malformed storage
bindings have individual `_fail.llzk` diagnostic tests.

Aggregate copy boundaries each have a small `aggregate_*_pass.llzk` fixture.
Constant-global copying and `scf.execute_region` execution have separate tests
under `test/Witgen/`. Private-only circuits retain their visibility. Existing
preservation tests compare compute bodies and verify straight-line constrain
structure without requiring the entire module to remain unchanged.

Repeated degree lowering records the actual degree bound, so a later degree-two
pass can introduce additional auxiliaries without reusing existing member names.
