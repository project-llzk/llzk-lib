# Direct R1CS implementation review — round 2

Reviewed September 27, 2026, against the working tree based on
`ad76a9f1389692b3f5a64c9ba1e9afc18f1cb257`. This is a follow-up to
[the first review](direct-r1cs-implementation-review.md), including the fixes and
regressions added since that review. Implementation and tests were left unchanged.

## Findings

### DR-004 — P1: aggregate member reads alias storage and can produce an invalid witness

**Location:** `tools/llzk-witgen/Interpreter.cpp:712–725`, particularly the
`bind({it->second})` at line 725.

Reading an array-valued struct member returns the same shared aggregate storage.
Mutating the read result therefore changes the original member. This contradicts
`include/llzk/Dialect/Struct/IR/Ops.td:285–286`, which explicitly specifies a value
copy rather than an alias. The new recursive `copyArgument` fixes function-call
boundaries, but does not cover this operation.

A valid compute function stores `[x]` in a member, reads a copy, writes zero to
that copy, then reads the original member into `out`. Its constraint is `out = x`.
For input `[3]`, value semantics require `out = 3`. The interpreter instead returns
`out = 0`. Both binary export and witness generation succeed, but the emitted
WTNS values are `[1, 0, 3]`. Independently decoding the R1CS gives evaluated
constraint sides `A = 0, B = 0, C = 3`; thus `A * B != C` modulo 2013265921.

This is an inherited interpreter defect, not a regression introduced by
`copyArgument`. It remains relevant to the direct pipeline because retained
compute bodies execute through this interpreter. The first-round mutable-input
regressions exercise function argument copying, not aggregate member reads.

**Recommended fix:** apply the documented copy semantics at aggregate read
boundaries, using a shared recursive value-copy helper. Audit aggregate writes,
POD reads/writes, array element reads/writes, and array construction/extract/insert
for equivalent shallow copies; those sites also store or return shared aggregate
values. Only the struct-member read case is experimentally confirmed here.
Add a regression that mutates a read copy and checks both the original storage
and every binary R1CS equation against its WTNS witness.

**Standalone reproducer:** save as `member-copy.llzk`.

```mlir
module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {
  struct.def @Main {
    struct.member @values : !array.type<1 x !felt.type<"babybear">>
    struct.member @out : !felt.type<"babybear"> {llzk.pub}
    function.def @compute(%x: !felt.type<"babybear"> {llzk.pub}) -> !struct.type<@Main> {
      %self = struct.new : <@Main>
      %zero = arith.constant 0 : index
      %z = felt.const 0 : !felt.type<"babybear">
      %a = array.new %x : !array.type<1 x !felt.type<"babybear">>
      struct.writem %self[@values] = %a : !struct.type<@Main>, !array.type<1 x !felt.type<"babybear">>
      %copy = struct.readm %self[@values] : !struct.type<@Main>, !array.type<1 x !felt.type<"babybear">>
      array.write %copy[%zero] = %z : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
      %original = struct.readm %self[@values] : !struct.type<@Main>, !array.type<1 x !felt.type<"babybear">>
      %value = array.read %original[%zero] : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
      struct.writem %self[@out] = %value : !struct.type<@Main>, !felt.type<"babybear">
      function.return %self : !struct.type<@Main>
    }
    function.def @constrain(%self: !struct.type<@Main>, %x: !felt.type<"babybear"> {llzk.pub}) {
      %out = struct.readm %self[@out] : !struct.type<@Main>, !felt.type<"babybear">
      constrain.eq %out, %x : !felt.type<"babybear">
      function.return
    }
  }
}
```

Run within the Nix development environment, with freshly built tools:

```sh
build/bin/llzk-opt member-copy.llzk --llzk-monomorphize --llzk-evaluate-constraints -o member-copy.eval.llzk
build/bin/llzk-translate member-copy.eval.llzk --llzk-to-r1cs --r1cs-prime=2013265921 -o member-copy.r1cs
printf '[3]\n' > member-copy.json
build/bin/llzk-witgen member-copy.eval.llzk --inputs member-copy.json --output-wtns member-copy.wtns
```

## First-round follow-up

- **DR-001:** entry-value capture is now present in both auxiliary reconstruction
  paths, and interpreter function arguments are recursively copied. The new
  mutable-input, helper-input, multidimensional-input, and R1CS-auxiliary tests
  pass. DR-004 identifies a separate remaining copy boundary.
- **DR-002:** direct lowering now derives storage paths from the actual SSA read
  chain and compares supplied annotations against them. Invalid-binding
  regressions pass.
- **DR-003:** normalization and equation lowering propagate failures instead of
  continuing to an unreachable path. The unsupported-division regression passes.

No additional defect was confirmed in those three fixes during this pass.

## Validation

Command:

```sh
nix develop .#release --offline --command bash -c 'cmake --build build --target check -j 8'
```

- Lit: **476 passed**, 4 unsupported, 1 expected failure; no unexpected failures.
- Unit/C API: **1,342 passed**, zero failures.
- Generated **40 deterministic polynomial DAGs** with seed 7921, each containing
  12 add/subtract/multiply/negate expressions over two BabyBear inputs. Each case
  ran monomorphization, symbolic evaluation, direct binary R1CS export, and WTNS
  generation at inputs `[2, 3]`. An independent Python decoder checked every
  constraint modulo the binary's prime and compared the output with Python field
  arithmetic. **All 40 passed.** This is sampled coverage, not exhaustive testing
  of fields, control flow, or aggregate combinations.
- The separate aggregate-copy reproducer completed successfully at the tool
  level but failed its independently checked binary constraint as described above.

Temporary evidence is in `/tmp/llzk-direct-review-2/`: `check.log`, `probe.py`,
`probe.log`, `dag-results.json`, generated source/evaluated IR, and binary artifacts.
The full confirmed reproducer is embedded above so it survives temporary cleanup.

No performance measurements were taken in this round. This review does not
establish complete aggregate copy correctness or comprehensive diagnostic coverage.

## Follow-up fix in the originating session

**DR-004 addressed.** The interpreter's recursive runtime-value copy helper is
now used at struct member reads and writes, POD reads/writes and initialization,
array element reads/writes and initialization, and array extraction/insertion,
as well as function argument boundaries. Mutation targets still use their own
storage; transferred aggregate values no longer share nested mutable storage.

`test/Transforms/R1CSLowering/aggregate-value-copies.llzk` contains ten cases,
one per aggregate boundary. Its Python driver evaluates and exports each case,
runs three input values, verifies the output and witness wire values, and checks
every binary R1CS equation against the WTNS witness. This includes the confirmed
struct-member read reproducer. It is focused regression coverage, not a claim of
exhaustive aggregate or control-flow testing.

Validation: clang-format verification and the Nix `check` target passed.
**477 lit tests passed**, 4 unsupported, 1 expected failure (482 discovered), and
**all 1,342 unit/CAPI tests passed**. Log: `/tmp/llzk-aggregate-copy-check.log`.

### Subsequent test organization

The combined aggregate fixture was replaced by ten independent
`test/Transforms/R1CSLowering/aggregate_*_pass.llzk` files, each using one nonzero
input. Diagnostics use separate `_fail.llzk` files. Other combined regressions
were likewise split by behavior; numerical R1CS/WTNS agreement remains checked.
After this restructuring, 506 lit tests and 1,350 unit/CAPI tests pass.
