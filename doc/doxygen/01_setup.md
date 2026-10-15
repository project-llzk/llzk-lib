# Setup {#setup}

\tableofcontents

There are two options for setting up the environment:
* Nix (recommended)
* manual (does not require Nix)

Tip: Following the manual setup steps through cloning the LLVM project in the
`third-party` directory may enable code exploration in your IDE.

# Nix Setup

This repository is already configured with a Nix flakes environment. The flake
was built using nix version 2.24.11 and may not be compatible with newer versions,
so if you encounter issues, consider downgrading/upgrading to 2.24.

To use the LLZK derivation to build and test the project, you can run `nix build`
from the repository root (add `-L` if you want to print the logs while building).

Alternatively, to launch a developer shell, run the following command:

```bash
nix develop
```

Within the developer shell, run the following command to generate the CMake configuration:

```bash
phases=configurePhase genericBuild
```
or (since v23.11)
```bash
runPhase configurePhase
```

By default, the developer shell is set up to build in debug mode. If you want to
generate a release build, append `-DCMAKE_BUILD_TYPE=Release` to `cmakeFlags`:

```
phases=configurePhase cmakeFlags="$cmakeFlags -DCMAKE_BUILD_TYPE=Release" genericBuild
```

Notes:

* Nix flakes are required for this to work.
* Nix 2.13 is assumed. Compatibility with other versions has not been checked
  yet, but they should work.

# Python Bindings

The Python-enabled package uses the Python, nanobind, and LLVM/MLIR versions
from the pinned Nix dependencies. Build and test both the build-tree and
installed bindings with:

```sh
nix build -L .#llzk-with-python
nix build -L .#checks.aarch64-darwin.llzk-installcheck-python
```

Use `x86_64-linux` instead of `aarch64-darwin` on Linux. The normal LLZK build
does not enable Python; its passing tests alone do not validate these bindings.

For development, use the dedicated shell and a separate build directory:

```sh
nix develop .#python
cmake -S . -B build-python -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DLLZK_ENABLE_BINDINGS_PYTHON=ON -DLLZK_BUILD_DEVTOOLS=ON \
  -DLLZK_VERSION_OVERRIDE=3.0.0
cmake --build build-python --target check-lit
```

The shell puts `build-python/python` and the matching installed MLIR package on
`PYTHONPATH`. LLZK installs its packages under `<prefix>/python`; add that
directory to `PYTHONPATH` to use an installed build with the same MLIR environment.
Outside Nix, install nanobind and MLIR's Python dependencies, build MLIR with
`MLIR_ENABLE_BINDINGS_PYTHON=ON`, and configure LLZK against that installation.
The MLIR Python package must be importable by the configured Python interpreter.

```python
from mlir import ir
from mlir.dialects import llzk as core
import llzk

with ir.Context() as context, ir.Location.unknown():
    llzk.add_dialects_to_context(context)
    context.load_all_available_dialects()
    felt_type = ir.Type.parse("!felt.type")
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
        core.NonDetOp(felt_type)
    assert module.operation.verify()
```

Operation classes are generated from TableGen for the `array`, `bool`, `cast`,
`constrain`, `felt`, `function`, `global`, `include`, `llzk`, `pod`, `poly`, `ram`,
`smt_info`, `string`, `struct`, and `verif` namespaces. Import them from
`mlir.dialects.<namespace>`; use `importlib.import_module("mlir.dialects.global")`
for the Python keyword `global`. The earlier `mlir.dialects.llzk_dialect` import
remains an alias for the core `llzk` module.

Custom types and attributes support generic `ir.Type.parse` and
`ir.Attribute.parse`, printing, equality, and use in generated builders. Dedicated
Python subclasses and convenience constructors for every custom type/attribute
are not provided. Backend dialect wrappers are also outside this package's
current scope.

For explicit registration, create an `ir.DialectRegistry`, call
`llzk.registration.register_dialects(registry)`, and append it to a context.
Call `llzk.registration.register_passes(registry)` once before creating pass
pipelines to register core LLZK passes and their context extensions. Append that
registry to the context used by the pass manager. Registration makes dialects
available for lazy loading during parsing; call
`context.load_all_available_dialects()` before constructing operations directly
or querying operation registration. Use
`llzk-remove-dead-values` for LLZK's poison-cleanup variant of MLIR's pass;
MLIR's Python package already registers the upstream `remove-dead-values` name.
Native LLZK tools retain that older name as an alias for the LLZK variant.

# Manual Build Setup

Note that the manual build instructions may be out of date; if you encounter issues, look at the build options as configured in Nix.
LLZK requires the following to be installed:

* CMake 3.18 or newer
* Ninja
* Z3
* Clang 16 or higher (use the same compiler for both LLVM and LLZK repos)

To optionally generate API documentation, you need:
* Doxygen (tested on 1.10 and newer)

To run tests, you also need:
* Python 3
* llvm-lit
* gtest

Note that tests are enabled by default; they can be disabled by setting
`-DBUILD_TESTING=off` when invoking CMake.

Once you have CMake, Ninja, and Python3, you can use the following script to
build the rest of the dependencies and LLZK:

```bash
# Start from llzk repo top level.

# LLVM's install tree must be outside the LLZK source tree.
export INSTALL_ROOT="$(dirname "$(pwd -P)")/llzk-llvm-install-root"
mkdir -p "$INSTALL_ROOT"

# First, build LLVM + MLIR
rm -rf third-party
mkdir third-party
pushd third-party
export THIRD_PARTY="$PWD"

# Configure LLVM build
git clone https://github.com/llvm/llvm-project.git -b llvmorg-23.1.3 --depth 1
pushd llvm-project
mkdir build
pushd build
cmake ../llvm -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_INCLUDE_BENCHMARKS=off \
  -DLLVM_INCLUDE_EXAMPLES=off \
  -DLLVM_BUILD_TESTS=off \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_ROOT" \
  -DLLVM_ENABLE_RTTI=on \
  -DLLVM_ENABLE_EH=on \
  -DLLVM_ENABLE_ASSERTIONS=on \
  -DLLVM_INSTALL_UTILS=on \
  -DLLVM_ENABLE_Z3_SOLVER=on

# Build LLVM (note that this will take a while, around 10 minutes on a Mac M1)
cmake --build .
cmake --build . --target install
popd # back to llvm-project
popd # back to third-party
popd # back to top level

export LIT_PATH=$PWD/third-party/llvm-project/llvm/utils/lit/lit.py

# Generate LLZK build configuration.
# You can set BUILD_TESTING=off if you don't want to enable tests.
# You can set CMAKE_BUILD_TYPE=DebWithSans if you want to enable sanitizers.
rm -rf build
mkdir build
pushd build
cmake .. -GNinja \
  -DLLVM_ROOT="$INSTALL_ROOT" \
  -DLLVM_DIR="$INSTALL_ROOT"/lib/cmake/llvm \
  -DMLIR_DIR="$INSTALL_ROOT"/lib/cmake/mlir \
  -DClang_DIR="$INSTALL_ROOT"/lib/cmake/clang \
  -DLLVM_EXTERNAL_LIT="$LIT_PATH" \
  -DLLZK_BUILD_DEVTOOLS=ON
```

# Development Workflow {#dev-workflow}

Once you have generated the build configuration and are in the `build` directory,
you can run the following commands:

* Compile: `cmake --build .`
* Run all tests: `cmake --build . --target check`
  * To run only unit tests: `cmake --build . --target check-unit`
  * To run only lit tests: `cmake --build . --target check-lit`
* Generate API docs (in `doc/html`): `cmake --build . --target doc`
* Run install target (requires `CMAKE_INSTALL_PREFIX` to be set):
  `cmake --build . --target install`
* Run clang-format on C++ and tablegen files:
  ``clang-format -i $(find include lib tools backends unittests -name '*.h' -o -name '*.cpp' -o -name '*.td' -type f)``
* Run clang-tidy on C++ files:
  ``clang-tidy -p build/compile_commands.json $(find include/llzk lib tools backends unittests -name '*.h' -o -name '*.cpp' -type f)``

The build configuration will automatically export `compile_commands.json`, so
LSP servers such as `clangd` should be able to pick up helpful IDE information
like include paths, etc.

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref overview | \ref tools |
</div>
