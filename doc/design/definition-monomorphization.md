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

The installed frontend is Circom 2.2.2 at
`/Users/shankarapailoor/veridise/circom/result/bin/circom`.
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

### Release measurements

Measured on macOS arm64, 2026-09-26, with a clean source tree at
`cb55838f0c86fcbb2fba174cb7629b398f7b0120`. These are single samples, not statistical
performance claims. The compiler build was idle during measurement. Peak memory is the
llzk-opt process high-water mark reported by `/usr/bin/time -l`.

| Input | Frontend mode | Definitions | Input ops | Specialized ops | Total output ops | Pass ms | Process wall s | Peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| and | templated | 1 | 14 | 12 | 26 | 0.056 | 0.037 | 22.25 |
| iszero | templated | 1 | 34 | 32 | 66 | 0.174 | 0.024 | 22.50 |
| num2bits | templated | 1 | 60 | 57 | 117 | 1.310 | 0.024 | 23.14 |
| switcher | templated | 1 | 30 | 28 | 58 | 0.045 | 0.012 | 22.27 |
| sigma | templated | 1 | 26 | 24 | 50 | 0.046 | 0.023 | 22.22 |
| binsum | concrete | 1 | 117 | 115 | 232 | 0.090 | 0.022 | 22.75 |
| poseidon3 | concrete | 75 | 339,150 | 339,070 | 678,220 | 164.031 | 2.625 | 291.66 |
| poseidon6 | concrete | 78 | 343,149 | 343,066 | 686,215 | 164.883 | 2.713 | 298.14 |

Poseidon3 frontend emission took 1.182 s and peaked at 304.39 MiB; Poseidon6 took
1.227 s and peaked at 301.62 MiB. These costs are separate from the pass/process
columns above. The source already contains roughly 339–343 thousand operations; this
checkpoint retains originals and adds one clone per reachable concrete definition.
It does not yet trim unused originals or compress frontend constant-building code.
The focused reuse test verifies that ten loop iterations and repeated calls still use
one definition per tuple; the family test discovers three definitions for indices
0, 1, 2 while preserving its single source loop.

The final sample commands, run inside the Release Nix development environment, were:

```sh
python3 scripts/benchmark-monomorphization.py \
  --corpus /Users/shankarapailoor/veridise/circom-benchmarks \
  --frontend /Users/shankarapailoor/veridise/circom/result/bin/circom \
  --llzk-opt build/bin/llzk-opt --jobs 1 --build-type Release \
  --tier small --output /tmp/llzk-mono-final-small

python3 scripts/benchmark-monomorphization.py \
  --corpus /Users/shankarapailoor/veridise/circom-benchmarks \
  --frontend /Users/shankarapailoor/veridise/circom/result/bin/circom \
  --llzk-opt build/bin/llzk-opt --jobs 1 --build-type Release \
  --frontend-mode concrete --plaintext --filter 'binsum|poseidon' \
  --output /tmp/llzk-mono-final-scale
```

Frontend IR and native R1CS files were temporary and removed automatically. Summaries
above retain the useful measurements; raw benchmark summaries, logs, and transient
fixtures were removed after summarization and are not committed.

### Final build validation

`nix build -L` succeeded for implementation revision
`cb55838f0c86fcbb2fba174cb7629b398f7b0120`, producing the Release 3.0.0 package.
Its checks passed all 1,340 unit/C API tests and the same 453-test lit suite
(448 passed, four unsupported, one expected failure). The standalone CMake
`check-lit` target also passed. No historical performance-branch changes were imported.
