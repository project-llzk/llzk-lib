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

# Manual Build Setup

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

# First, build LLVM + MLIR
mkdir third-party
pushd third-party
export THIRD_PARTY="$PWD"

# This is where llvm will be installed.
export INSTALL_ROOT="$THIRD_PARTY/llvm-install-root"
mkdir "$INSTALL_ROOT"

# Build LLVM (note that this will take a while, around 10 minutes on a Mac M1)
git clone https://github.com/llvm/llvm-project.git -b llvmorg-20.1.8 --depth 1
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
  -DLLVM_BUILD_LLVM_DYLIB=on \
  -DLLVM_LINK_LLVM_DYLIB=on \
  -DLLVM_ENABLE_RTTI=on \
  -DLLVM_ENABLE_EH=on \
  -DLLVM_ENABLE_ASSERTIONS=on \
  -DLLVM_INSTALL_UTILS=on \
  -DLLVM_ENABLE_Z3_SOLVER=on

# Note that using llvm dylib will cause llzk to be linked to the built LLVM
# dylib; if you'd like llzk to be used independent of the build folder, you
# should leave off the dylib settings.

cmake --build .
cmake --build . --target install
popd # back to llvm-project
popd # back to third-party
popd # back to top level

# Use an alias to avoid "prefixed in the source directory" CMake error.
ln -s $INSTALL_ROOT ~/llvm-install-root-llzklib
export INSTALL_ROOT=~/llvm-install-root-llzklib
export LIT_PATH=$PWD/third-party/llvm-project/llvm/utils/lit/lit.py

# Generate LLZK build configuration.
# You can set BUILD_TESTING=off if you don't want to enable tests.
# You can set CMAKE_BUILD_TYPE=DebWithSans if you want to enable sanitizers.
mkdir build && cd build
cmake .. -GNinja \
  -DLLVM_ROOT="$INSTALL_ROOT" \
  -DLLVM_DIR="$INSTALL_ROOT"/lib/cmake/llvm \
  -DMLIR_DIR="$INSTALL_ROOT"/lib/cmake/mlir \
  -DClang_DIR="$INSTALL_ROOT"/lib/cmake/clang \
  -DLLVM_EXTERNAL_LIT="$LIT_PATH" \
  -DGTEST_ROOT="$INSTALL_ROOT" \
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
  * Note that due to bugs in clang-tidy, this may segfault if running on all files.

The build configuration will automatically export `compile_commands.json`, so
LSP servers such as `clangd` should be able to pick up helpful IDE information
like include paths, etc.

<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref overview | \ref tools |
</div>
