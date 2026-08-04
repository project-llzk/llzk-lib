# AGENTS.md

## Build and command guidance

- Preferred build command: `nix build -L`
- When you need to run a project command inside the development environment, use:
  `nix develop --command bash -c "[command]"`

### Tool freshness

- After modifying C++/TableGen/build files that can affect `llzk-opt`, do not run `llzk-opt` or any example pipeline using `llzk-opt` until rebuilding it first.
- Preferred command before any manual `llzk-opt ...` invocation after relevant edits:
  `nix develop --command bash -c "cmake --build build --target llzk-opt"`
- When validating lit/FileCheck tests, prefer `cmake --build build --target check-lit` rather than manually running a possibly stale `llzk-opt`; the lit config resolves tools from the build tree.
- If a manual `llzk-opt` command fails unexpectedly after code changes, rebuild `llzk-opt` once before debugging the behavior.

### Lit test execution

- Do not run `llvm-lit` directly unless you have first confirmed it is available in the active environment.
- Prefer CMake test targets because they use the lit runner and freshly built tools from the configured build tree:
  `nix develop --command bash -c "cmake --build build --target check-lit"`
- For focused lit coverage, prefer an appropriate CMake/ninja target over invoking `llvm-lit` from the ambient shell. If a direct lit command is unavoidable, run it inside the Nix development environment and use the configured lit runner from the build tree rather than assuming `llvm-lit` is on `PATH`.
- If lit/FileCheck output looks inconsistent with recent C++/TableGen/build changes, rebuild the affected tool target first, especially `llzk-opt`, before debugging test behavior:
  `nix develop --command bash -c "cmake --build build --target llzk-opt"`

## Fast context for agents

- For project questions and code-change tasks, start by checking the generated project documentation in `doc/doxygen/`.
- Prefer searching documentation and targeted source paths before broad repo scans to reduce turnaround time.
- When touching tests that use FileCheck, generate or refresh check lines with `scripts/generate-test-checks.py` instead of hand-editing check blocks. Use a pipeline such as: `nix develop --command bash -c "llzk-opt test/your_test.llzk <passes> | scripts/generate-test-checks.py"`.

## FileCheck update safety

- Never run `scripts/generate-test-checks.py --source <file> -i` as the first attempt on a repo file.
- First run the script without `-i`, writing output only to stdout or a temporary file, and confirm it succeeds. Never use `-o` or shell redirection to write a dry run back to the repo file.
- For `-split-input-file` tests or files with many existing CHECK blocks, prefer generating checks from a temporary file containing only the changed source chunks.
- When using `--source_delim_regex`, choose a regex that matches the actual chunk boundaries in the source file for that run. Anchor it to real source lines so it does not match existing `// CHECK` comments.
- Do not assume `^module attributes` is correct for a whole `-split-input-file` test; it is mainly useful when working from a temporary file whose chunks each begin with a module line.
- Do not rely on the script's default delimiter regex for multi-module or split-input LLZK tests.
- If the correct delimiter is uncertain, extract only the changed source chunks into a temporary file and generate checks there instead of updating the full repo file in place.
- Only use `-i` after a non-inplace dry run has succeeded with the same arguments.

## Search boundaries

- Prefer searching these paths first: `doc/doxygen/`, `include/`, `lib/`, `tools/`, `test/`, `unittests/`, `backends/`.
- Avoid broad scans of generated or vendored trees unless the task specifically requires them: `build/`, `result/`, `third-party/llvm-project/`.

## Where to work

- Dialects and IR definitions: `include/llzk/Dialect/`, `lib/Dialect/`
- Transform passes and rewrite pipelines: `include/llzk/Transforms/`, `lib/Transforms/`
- Program analyses: `include/llzk/Analysis/`, `lib/Analysis/`
- Program validation: `include/llzk/Validators/`, `lib/Validators/`
- C API implementation and headers: `include/llzk-c/`, `include/llzk/CAPI/`, `lib/CAPI/`
- Shared helpers/utilities: `include/llzk/Util/`, `lib/Util/`
- CLI tools: `tools/`
- Backends: `backends/`
- Integration tests (lit/FileCheck): `test/`
- Unit tests: `unittests/`
- C API unit tests: `unittests/CAPI/`
- Python bindings and helpers: `python/`

## Deeper docs

- Project-wide generated docs: `doc/doxygen/`
- Project overview and usage: `README.md`
- ZKLean backend notes: `backends/zklean/ZKLean.md`
- Build/config entrypoints: `CMakeLists.txt`, `CMakePresets.json`, `flake.nix`

## Code change workflow

- For bug fixes or behavior changes, identify the affected component, update the implementation, and add or update focused tests.
- When a test only checks whether IR parsing or verification succeeds or fails, prefer a lit test under `test/` instead of a unit test under `unittests/`. Reserve unit tests for API-level assertions, structural inspection, or behaviors that lit coverage cannot express cleanly.
- When adding or generating new functions, classes, methods, passes, operations, or other non-trivial symbols, include appropriate documentation as part of the same change. Prefer Doxygen-style comments for declarations and keep implementation comments concise and behavior-focused.
- Keep diffs minimal and aligned with existing style and architecture.
- Validate changes with relevant build/test commands in the Nix environment.

## Quick verification

- Full build: `nix build -L`
- Lit tests: `nix develop --command bash -c "cmake --build build --target check-lit"`
- Unit tests: `nix develop --command bash -c "cmake --build build --target check-unit"`
- All tests: `nix develop --command bash -c "cmake --build build --target check"`

## Completion requirement

- When modified files affect the build, ensure the build is successful before you stop working.
