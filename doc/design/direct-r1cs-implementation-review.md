# Direct R1CS implementation review

Review date: 2026-09-27
Reviewed checkout: `/Users/shankarapailoor/.codex/worktrees/27e4/llzk-lib`
Base: `ad76a9f1389692b3f5a64c9ba1e9afc18f1cb257` (`HEAD`)

## Scope and outcome

Reviewed the accumulated staged and unstaged changes against `HEAD`, plus the four untracked `evaluated-*.llzk` R1CS tests. Scope included direct symbolic evaluation, cleanup, storage identities and visibility, degree and R1CS auxiliary generation, the in-memory translation entry point, binary/witness wire ordering, APInt/GMP folding and inversion caching, build dependencies, MLIR pass conventions, and regression coverage. Existing binary exporter and interpreter code was also traced where the new paths depend on it. Benchmark reports were treated as prior measurements, not independently reproduced performance evidence.

Three actionable findings are confirmed below. DR-001 affects ordinary supported input and produces an invalid witness without an error. DR-002 requires inconsistent binding metadata; it is a validation gap, not a claim that the evaluator currently produces such metadata. DR-003 is an inherited lowering error-handling weakness exposed by the new integrated entry point. These distinctions matter when choosing fixes and release scope.

| ID | Priority | Finding |
|---|---|---|
| DR-001 | P1 — high | Auxiliary reconstruction rereads mutable input storage after it has changed |
| DR-002 | P2 — medium | Binding metadata can merge distinct actual storage reads without a diagnostic |
| DR-003 | P2 — medium | Unsupported division continues past pass failure and traps in the direct exporter |

No implementation fixes or existing-test edits were made. This review file is the sole intended repository change from the review session. Temporary inputs and logs are under `/tmp/llzk-direct-review/`; essential reproduction details are included below so findings do not depend on those files surviving.

## Confirmed findings

### DR-001 — Preserve entry input values when rebuilding auxiliary computations

**Priority:** P1 — high. **Status:** reproduced with binary R1CS and WTNS, not just text inspection.

**Locations:**

- [lib/Transforms/LLZKLoweringUtils.cpp](../../lib/Transforms/LLZKLoweringUtils.cpp), lines 43–49 and 77–90: constrain arguments map to compute arguments, and array/POD reads are cloned at the caller's insertion point.
- [lib/Transforms/LLZKPolyLoweringPass.cpp](../../lib/Transforms/LLZKPolyLoweringPass.cpp), lines 1003–1023: that insertion point is immediately before each compute return.
- [backends/r1cs/lib/r1cs/Transforms/R1CSLoweringPass.cpp](../../backends/r1cs/lib/r1cs/Transforms/R1CSLoweringPass.cpp), lines 920–938: R1CS normalization auxiliaries use the same late reconstruction strategy.
- Related existing interpreter behavior: [tools/llzk-witgen/WitgenDriver.cpp](../../tools/llzk-witgen/WitgenDriver.cpp), lines 132–148, serializes the argument values after interpretation; shared aggregate storage lets mutations affect the reported input too.

**Trigger and impact:** A legal compute method reads an array input, calculates its output, and then overwrites its local input array before returning. Its constrain method expresses the output in terms of the original input. Function arguments are documented as passed by value in [include/llzk/Dialect/Function/IR/Ops.td](../../include/llzk/Dialect/Function/IR/Ops.td), lines 52–55. This source pattern must therefore not change the caller's input.

The new direct pipeline keeps the mutation in rolled compute. When degree lowering introduces an auxiliary for the original input's square, it appends a fresh array read **after the overwrite**. That auxiliary no longer describes the constraint input. The witness tool returns success and writes a `.wtns` that fails the emitted R1CS.

**Minimal reproduction:** Save the following as `/tmp/llzk-direct-review/mutated-input.llzk`:

```mlir
module attributes {llzk.lang, llzk.main = !struct.type<@Main>} {
  struct.def @Main {
    struct.member @out : !felt.type<"babybear"> {llzk.pub}
    function.def @compute(%a: !array.type<1 x !felt.type<"babybear">> {llzk.pub}) -> !struct.type<@Main> {
      %s = struct.new : <@Main>
      %i = arith.constant 0 : index
      %x = array.read %a[%i] : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
      %x2 = felt.mul %x, %x : !felt.type<"babybear">, !felt.type<"babybear">
      %x3 = felt.mul %x2, %x : !felt.type<"babybear">, !felt.type<"babybear">
      struct.writem %s[@out] = %x3 : !struct.type<@Main>, !felt.type<"babybear">
      %z = felt.const 0 : !felt.type<"babybear">
      array.write %a[%i] = %z : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
      function.return %s : !struct.type<@Main>
    }
    function.def @constrain(%s: !struct.type<@Main>, %a: !array.type<1 x !felt.type<"babybear">> {llzk.pub}) {
      %i = arith.constant 0 : index
      %x = array.read %a[%i] : !array.type<1 x !felt.type<"babybear">>, !felt.type<"babybear">
      %x2 = felt.mul %x, %x : !felt.type<"babybear">, !felt.type<"babybear">
      %x3 = felt.mul %x2, %x : !felt.type<"babybear">, !felt.type<"babybear">
      %out = struct.readm %s[@out] : !struct.type<@Main>, !felt.type<"babybear">
      constrain.eq %out, %x3 : !felt.type<"babybear">
      function.return
    }
  }
}
```

From the repository root, run inside `nix develop .#release --offline --command bash`:

```sh
build/bin/llzk-opt /tmp/llzk-direct-review/mutated-input.llzk \
  --llzk-monomorphize --llzk-evaluate-constraints \
  -o /tmp/llzk-direct-review/mutated-input.eval.llzk
printf '[[3]]\n' > /tmp/llzk-direct-review/input.json
build/bin/llzk-witgen /tmp/llzk-direct-review/mutated-input.eval.llzk \
  --inputs /tmp/llzk-direct-review/input.json \
  --output-wtns /tmp/llzk-direct-review/mutated-input.wtns
build/bin/llzk-translate /tmp/llzk-direct-review/mutated-input.eval.llzk \
  --llzk-to-r1cs --r1cs-prime=2013265921 \
  -o /tmp/llzk-direct-review/mutated-input.r1cs
```

**Observed evidence:**

- The computed public output remains `27`.
- Reported input is `["0"]`, and `__llzk_poly_lowering_pass_aux_member_0` is `"0"`, although the entry input was `3` and its square is `9`.
- Binary WTNS values are `[1, 27, 0, 0]`, ordered as constant-one, output, input, auxiliary.
- Independently decoding both binary files gives equations `(0 * 0 = 0)` and `(0 * 0 = 27)`; the second fails modulo `2013265921`.
- Lowered compute contains `array.write %arg0[...] = 0`, followed by a new `array.read %arg0[...]`, then the generated auxiliary assignment.

The input-serialization alias is existing interpreter behavior, but fixing that alone is insufficient: a correct pass-by-value interpreter would still compute the newly appended auxiliary from its mutated local array. The new direct aggregate-read reconstruction needs to preserve entry values too.

**Suggested direction:** Capture the required input leaves or an independent aggregate snapshot at compute entry and use those values when rebuilding input-rooted expressions. Continue reading output-rooted paths from the actual returned component. Until mutation/aliasing is supported soundly, diagnose such inputs instead of returning a bad witness. Ensure the interpreter preserves argument value semantics and serializes the original inputs. Add an end-to-end binary satisfaction test for the reproduction, with analogous POD/subarray coverage.

### DR-002 — Validate binding identity against the actual storage read

**Priority:** P2 — medium. **Status:** reproduced; requires inconsistent or stale `poly.signal_binding` metadata.

**Locations:**

- [backends/r1cs/lib/r1cs/Transforms/R1CSLoweringPass.cpp](../../backends/r1cs/lib/r1cs/Transforms/R1CSLoweringPass.cpp), lines 649–655: the annotated path is accepted after shape and root-argument checks.
- Same file, lines 582–589: values with equal annotation paths are assigned one signal, independent of their real operands.
- [include/llzk/Dialect/Polymorphic/Transforms/ConstraintEvaluation.h](../../include/llzk/Dialect/Polymorphic/Transforms/ConstraintEvaluation.h), lines 35–57: `getStorageBinding` validates only attribute structure, not the referenced storage location.

**Trigger and impact:** A read of `children[1].value` is annotated with the otherwise well-formed path for `children[0].value`. Input verification and lowering both succeed. The second location disappears from the circuit and both read values use the first location's wire. This changes the meaning of the actual LLZK read operations rather than diagnosing malformed metadata.

The tests already state that metadata is user-editable and malformed bindings should diagnose, in [test/Transforms/SymbolicEvaluation/check-direct-pipeline.py](../../test/Transforms/SymbolicEvaluation/check-direct-pipeline.py), lines 51–64. Their current cases check attribute type, empty/negative paths, root range, and Boolean type, but not consistency with the read.

**Reproduction:** Make a temporary copy of `test/Transforms/R1CSLowering/evaluated-read-identity.llzk`, omitting the generated checks. Change only this attribute on `%b = struct.readm %child1[@value]`:

```mlir
// Actual read still uses %child1, but the annotation claims child0:
{poly.signal_binding = {path = [0, "children", 0 : index, "value"], public = false}}
```

Run:

```sh
build/bin/llzk-opt /tmp/llzk-direct-review/mismatched-binding.llzk \
  --llzk-full-r1cs-lowering \
  -o /tmp/llzk-direct-review/mismatched-binding.r1cs.llzk
```

**Observed evidence:** The wire map contains only `children[0].value`; there is just one internal `r1cs.def`. The two original equations `a*a = x` and `a*b = y` become `a*a = x` and `a*a = y`. For example, `a=2, b=3, x=4, y=4` satisfies the emitted equations but violates the original second equation.

This is not evidence of spontaneous incorrect annotations from the current evaluator. It is a confirmed lack of verification at an input boundary that explicitly accepts already evaluated modules, and it can also conceal stale annotations after another transformation.

**Suggested direction:** Reconstruct or verify storage paths from the read's def-use chain, including root argument, member names, static indices and bounds, final felt type, and visibility. Intern the verified canonical path. If annotations are intended to be authoritative semantic data, give them a verifier-enforced contract rather than treating arbitrary dictionaries as proof of identity. Add mismatch, invalid-member/index, and conflicting-visibility diagnostics alongside the existing malformed-shape tests.

### DR-003 — Stop lowering after an unsupported-operation failure

**Priority:** P2 — medium. **Status:** reproduced twice, including after rechecking tool build freshness.

**Locations:**

- [backends/r1cs/lib/r1cs/Transforms/R1CSLoweringPass.cpp](../../backends/r1cs/lib/r1cs/Transforms/R1CSLoweringPass.cpp), lines 883–913 and 949–952: callers continue constructing constraints and enter `buildEvaluatedR1CS` after normalization has signaled failure.
- Same file, lines 416–420: unsupported normalization operations call `signalPassFailure()` but the routine still returns through `rewrites[root]`.
- Same file, lines 456–458: `lowerPolyToR1CS` reaches `llvm_unreachable` on the surviving unsupported operation.
- [backends/r1cs/lib/r1cs/Target/TranslateRegistration.cpp](../../backends/r1cs/lib/r1cs/Target/TranslateRegistration.cpp), lines 49–56: the new direct translation path exposes this behavior to ordinary LLZK input.

**Trigger and impact:** A constrain expression contains a division with symbolic operands. Evaluation accepts and emits it; degree computation recognizes `DivFeltOp`, so the degree pass does not reject this example. R1CS normalization reports an unsupported operation but does not stop. The process then traps instead of returning an MLIR diagnostic/failure.

**Reproduction:** Copy the source portion of `test/Transforms/R1CSLowering/evaluated-scalar-interface.llzk` to a temporary file, replacing `felt.mul %secret, %visible` with `felt.div %secret, %visible` in compute and constrain. Run:

```sh
build/bin/llzk-translate /tmp/llzk-direct-review/division.llzk \
  --llzk-to-r1cs --r1cs-prime=2013265921 \
  -o /tmp/llzk-direct-review/division.r1cs
```

**Observed evidence:** Python `subprocess.run` reports return code `-5` (SIGTRAP on this host). Standard error starts with:

```text
Unhandled op in normalize ForR1CS: %0 = felt.div ...
Unhandled op in R1CS lowering: %0 = felt.div ...
```

The stack includes `lowerPolyToR1CS`, `buildEvaluatedR1CS`, and `lowerAndExportR1CS`. The implementation explicitly allows only a polynomial subset, so this finding does **not** require adding division support. It requires rejecting unsupported input without invoking undefined/unreachable behavior. The underlying normalization/error-handling pattern predates parts of this change; the added direct branch retains that weakness.

**Suggested direction:** Return `FailureOr<Value>` or another explicit failure result from normalization, interrupt the equality walk on failure, and return from the pass before rebuilding or emitting a circuit. Keep user-reachable unsupported operations out of `llvm_unreachable` paths. Add a negative lit test that checks the diagnostic and ordinary nonzero exit, including the integrated translator.

## Open questions and remaining coverage gaps

These are not additional confirmed defects:

- **Marker lifecycle and no-op passes.** `poly.degree_lowered`, `r1cs.prepared`, and `r1cs.main` are serialized completion markers. The early-return branches are genuine no-ops for unchanged generated IR, so preserving analyses there is reasonable. What invalidates these markers if an intervening pass changes a constrain method or removes/replaces the generated circuit? Document the permitted pipeline lifecycle and consider validating marker targets. Unchanged-text idempotence tests do not exercise intervening rewrites.
- **Field agreement at direct export.** The integrated translator still takes an explicit `--r1cs-prime` separately from LLZK felt field annotations. Decide whether this entry point must reject mismatches or mixed fields before folding and export. The binary-only exporter previously had no LLZK field information to validate; the new entry point does. No cross-field reproduction was used as a finding here.
- **Degree tightening with witness validation.** The checked-in Python regression runs degree three followed by full R1CS lowering, but discards that run's result (lines 28–30). Its subsequent binary witness assertions use the directly lowered input. Preserve automated numerical validation of the tightened path, not only the manually reported historical check.
- **Nested PODs, multidimensional arrays, and copy boundaries.** Add numerical R1CS/WTNS cases for POD leaves, array extraction/insertion, helper argument mutation, and source-input preservation. Current nested-array coverage is valuable but does not establish value-copy behavior, as DR-001 demonstrates.
- **Visibility edges.** Add binary/header checks for unused private inputs, unused public array outputs, and a public child field below a private parent. The current tests cover much of the map structure but not every combination with numerical witness output.
- **Family witnesses.** The design document accurately limits family/grid tests to evaluated IR and does not claim initialized heterogeneous-family witness support. End-to-end coverage is still needed before broadening that claim.
- **Generated addition checks.** `evaluated-array-storage.llzk` permits arbitrary SSA operands in one `r1cs.add` to tolerate term order. The companion numerical tests compensate for the covered fixture; do not count that FileCheck line alone as verification that both expected operands are used.
- **Witness documentation.** `doc/doxygen/14_wtns_format.md` still says aggregate and synthesized auxiliary witnesses are unsupported. That describes the old path and should distinguish it from the new evaluated-module path.

The APInt/GMP implementation, inverse-cache normalization/capacity policy, and dependency wiring received code review and existing-test execution. No separate actionable arithmetic defect was established. The review did not establish exhaustive arithmetic correctness or test cross-platform GMP packaging.

## Checks performed

1. Read repository guidance and generated project documentation before targeted source inspection.
2. Inspected `git diff HEAD` and relevant untracked R1CS tests, tracing unchanged utility/exporter/interpreter code where required by the new design.
3. Rechecked `llzk-opt`, `llzk-translate`, and `llzk-witgen` build targets in the Nix release development environment; Ninja initially reported no work to do.
4. Independently ran:

   ```sh
   nix develop .#release --offline --command bash -c \
     'cmake --build build --target check -j 8'
   ```

   Result: success. **470 lit tests passed**, 4 unsupported, 1 expected failure (475 discovered); **all 1,342 unit/CAPI tests passed**. Log: `/tmp/llzk-direct-review/check.log`.
5. Reproduced DR-001 through evaluation, degree/R1CS lowering, witness generation, binary translation, and an independent Python decoder checking every emitted binary equation. Files: `mutated-input.*`, `mutation.log`, `check-mutation.py`, `check-mutation.log` under the temporary review directory.
6. Reproduced DR-002 using a temporary mutation of the existing read-identity fixture. Output: `/tmp/llzk-direct-review/mismatched-binding.r1cs.llzk`.
7. Reproduced DR-003 and rechecked it after confirming tool freshness. Logs: `/tmp/llzk-direct-review/boundary.log`, `division-recheck.log`, and `division.stderr`.
8. Rejected a candidate non-signal integer-storage concern because `i64` struct members fail input verification; it is not a finding.

No full benchmark rerun, external Circom comparison, release-package rebuild, or implementation changes were needed for these review conclusions. Prior timing numbers and large-circuit correctness claims were not independently re-established by this review.

## Follow-up fixes in the originating session

All three confirmed findings have been addressed in the working tree:

- **DR-001:** Both degree and R1CS lowering capture auxiliary input leaves at
  compute entry and reuse those scalar values at each return. Output-rooted
  expressions still read the returned component. The interpreter recursively
  copies aggregate arguments at function entry, preserving caller storage and
  the original inputs serialized into the witness. Binary R1CS/WTNS regressions
  exercise direct mutation, multidimensional arrays, helper argument mutation,
  and auxiliaries introduced by either lowering pass. They test three input
  values and both direct degree-two lowering and degree-three-to-two tightening.
- **DR-002:** R1CS lowering reconstructs canonical paths from the actual read
  chain, checks constant array indices and bounds, and derives visibility from
  the argument and member declarations. It rejects annotations with different
  paths or visibility. Diagnostic tests cover a different array element, an
  invalid index, an invalid member, and conflicting visibility. The positive
  identity test also checks equivalent integer encodings of the same path.
- **DR-003:** Normalization now returns an explicit failure and interrupts the
  equality walk before reconstruction or export. Polynomial conversion also
  propagates unsupported-operation failures rather than trapping. Regressions
  require ordinary nonzero exits and diagnostics from both `llzk-opt` and the
  integrated translator on symbolic division.

The WTNS documentation now distinguishes evaluated aggregate/auxiliary support
from the legacy scalar-only path. The open questions above remain follow-up
work; these fixes do not claim exhaustive coverage of every listed boundary.

Validation: formatting checks passed; the Nix `check` target built successfully,
with **476 lit tests passing**, 4 unsupported, 1 expected failure (481 discovered),
and **all 1,342 unit/CAPI tests passing**. Log:
`/tmp/llzk-review-fixes-final.log`.
The saved SHA2 input also exports a byte-for-byte identical R1CS binary to the
pre-fix reference (`/tmp/llzk-review-fixed-sha2.log`).
