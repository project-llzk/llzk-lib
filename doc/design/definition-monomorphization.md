# Definition monomorphization checkpoint

`llzk-opt input.llzk --llzk-monomorphize` is an opt-in, module-level pass rooted at
`llzk.main`. Existing flattening, scalarization, witness and R1CS pipelines are unchanged.
This is definition specialization, not instance elaboration or constraint evaluation.

## Registry and output contract

The registry uses `(source Operation*, ordered ArrayAttr arguments)` as its cache key.
Type arguments are structured `TypeAttr` entries. It reserves an ID before cloning and
processes newly discovered definitions with a growing worklist. It clones a whole
struct, including its methods, once per tuple; free functions receive their own entries.
Repeated calls or loop iterations do not clone bodies again. Non-parameterized reachable
definitions also receive identities. Originals are retained, including templates needed
by rolled heterogeneous calls. Consequently total output includes both retained source
and specialized definitions; use `specialized_operations` to isolate the latter.

Every generated definition preserves its source location and carries:

- `poly.specialization_id`: module-local integer identity.
- `poly.origin`: fully qualified original symbol.
- `poly.arguments`: ordered concrete parameters, including type arguments.

`include/llzk/Dialect/Polymorphic/Transforms/Specialization.h` exposes typed identity and
provenance readers. Generated names are references for MLIR symbol resolution; consumers
must not parse their spelling. IDs require remapping when independently processed modules
are merged. A second invocation on an already-specialized main does not create more clones.

Concrete type references and compute/constrain/free-function calls target the cached
definitions. Shared private constant-substitution helpers are reused from the flattening
implementation, but its driver and rewrite pipelines are never invoked. The separate
`DefinitionMonomorphization.inc` contains the new worklist driver.

Affine component families retain their original rolled types and call references:

- A member array with concrete shape records `poly.family`, an ordered array of
  `{indices, specialization}` dictionaries.
- A construction call records the unique candidate IDs in `poly.family_specializations`.
  Its original affine maps and SSA operands still describe which candidate applies.

This is a definition discovery contract. Instance IDs, access-path interning, dynamic
selection, and the Stage 2 descriptor API have not been implemented. No source loop or
function body is expanded. Static induction values are interpreted only for discovery.

## Supported boundary and diagnostics

Concrete struct parameters, explicit and signature-inferred free-function parameters,
type arguments, foldable template expressions, nested definitions, and scalar affine
parameters are supported. Array schemas support maps with one result and one input per
array dimension, including multidimensional shapes. Construction discovery supports
statically bounded `scf.for` loops with positive steps and no loop-carried arguments.
Ordinary `scf.for` and `scf.while` operations remain rolled.

Unsupported non-concrete parameters, unknown family indices, unsupported family maps,
external functions, non-compute/constrain struct methods, and family-producing loops outside that boundary
are diagnosed. Included definitions must be inlined before this pass. Changing-argument specialization recursion reports an ancestry chain;
call-graph cycles, including exact cache hits and affine candidates, are rejected too.
This deliberately rejects recursive definitions even when a particular execution might
terminate. `max-specializations` defaults to 10,000; affine discovery is capped at one
million visited indices/iterations. Discovery currently enumerates indices, rather than
using symbolic range compression.

The original templates remain required for heterogeneous call verification. Consumers
must use the family metadata and specialization IDs instead of interpreting the original
template body as already monomorphic. Nested type-argument families at construction sites,
loop-carried index evaluation, wildcard-body inference, and arbitrary struct methods need
further work. This checkpoint does not claim complete Stage 1 coverage for all frontend IR.

## Benchmark driver

Configure Release tools against Release dependencies, using the same version as the Nix
package when reading frontend bytecode:

```sh
nix develop .#release --command bash -c 'cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLZK_BUILD_DEVTOOLS=ON -DLLZK_VERSION_OVERRIDE=3.0.0'
nix develop .#release --command bash -c 'cmake --build build --target llzk-opt'
```

Use `scripts/benchmark-monomorphization.py` with `--corpus`, `--frontend`, `--llzk-opt`,
`--output`, `--build-type Release`, and `--jobs 1`. Select cases with `--filter` or `--tier small|scale`.
`--phase parse` measures loading alone; `--phase monomorphize` is the default.
`--frontend-mode templated|concrete` makes coverage explicit. `--plaintext` is an optional
frontend bytecode-compatibility fallback. The driver writes CSV/JSON summaries and logs
outside the repository, enforces timeouts, and deletes frontend IR after every case.
Processed IR is sent to `/dev/null`; there are no intermediate pass dumps.

The `report=true` pass option reports input operations, total output operations,
specialized operations, definition count, and pass-local milliseconds. The driver records
process wall time and peak RSS separately for frontend and llzk-opt. Pass-local time excludes
input parsing and pass-manager verification. Peak RSS includes parsing, retained originals,
clones, and verification; it is not a pass-exclusive memory measurement.

## Validation and measured coverage

Focused coverage includes reuse, distinct tuples, nested structs/free functions, explicit
and inferred type arguments, idempotence, affine member schemas and construction calls,
rolled `scf.for`/`scf.while`, and exact/changing-argument/affine recursion diagnostics.
The CMake `check-lit` run discovers 453 tests: 448 pass, four are unsupported, and one is
expected to fail.

The installed frontend is `/Users/shankarapailoor/veridise/circom/result/bin/circom`.
The corpus is `/Users/shankarapailoor/veridise/circom-benchmarks`. All runs are serial
(`--jobs 1`), with Release llzk-opt and Release LLVM/MLIR dependencies. Measurements use
`--llzk-monomorphize=report=true -o /dev/null`. Frontend time is recorded separately.
No PoseidonEx run was performed.

Templated AND, IsZero, Num2Bits, Switcher, and Sigma succeed. Templated BinSum is rejected
by the frontend (`function.call` in a `poly.expr` initializer); concrete BinSum succeeds.
Both templated Poseidon cases fail in the frontend with `variable 'nInputs' not found`.
Their concrete bytecode is rejected while parsing (`expected mlir::UnitAttr`), while
concrete plaintext is accepted. Those scale measurements therefore use `--plaintext`
with a temporary frontend input that is deleted at the end of each case. They demonstrate
registry/copy behavior on already-concrete frontend definitions, not successful templated
Poseidon specialization. Frontend compatibility needs resolving before that claim is made.

Fresh native Circom `--O0 --r1cs` baselines:

| Circuit | Constraints | Wires | Public outputs | Private inputs |
| --- | ---: | ---: | ---: | ---: |
| Poseidon3 | 765 | 768 | 1 | 2 |
| Poseidon6 | 1,347 | 1,353 | 1 | 5 |

Both have zero public inputs. This milestone does not lower the new representation to
R1CS or claim equation/witness equivalence; those checks belong to later stages.
