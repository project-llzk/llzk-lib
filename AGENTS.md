# AGENTS.md

## Build and command guidance

- Preferred build command: `nix build -L`
- When you need to run a project command inside the development environment, use:
  `nix develop --command bash -c "[command]"`

## Fast context for agents

- For project questions and code-change tasks, start by checking the generated project documentation in `doc/doxygen/`.
- Prefer searching documentation and targeted source paths before broad repo scans to reduce turnaround time.
- When touching tests that use FileCheck, generate or refresh check lines with `scripts/generate-test-checks.py` instead of hand-editing check blocks.
- For new checks from tool output, use a pipeline such as: `nix develop --command bash -c "llzk-opt test/your_test.llzk <passes> | scripts/generate-test-checks.py"`.
- For in-place check refresh in an existing test file, use: `nix develop --command bash -c "llzk-opt test/your_test.llzk <passes> | scripts/generate-test-checks.py --source test/your_test.llzk -i"`.

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
- Keep diffs minimal and aligned with existing style and architecture.
- Validate changes with relevant build/test commands in the Nix environment.

## Quick verification

- Full build: `nix build -L`
- Lit tests: `nix develop --command bash -c "cmake --build build --target check-lit"`
- Unit tests: `nix develop --command bash -c "cmake --build build --target check-unit"`
- All tests: `nix develop --command bash -c "cmake --build build --target check"`

## Completion requirement

- When modified files affect the build, ensure the build is successful before you stop working.
