#!/bin/bash

# Start from llzk repo top level.

# First, build LLVM + MLIR
rm -rf third-party
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
  -DLLVM_INSTALL_UTILS=on
  # -DLLVM_ENABLE_Z3_SOLVER=on

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
rm -rf build
mkdir build && cd build
cmake .. -GNinja \
  -DLLVM_ROOT="$INSTALL_ROOT" \
  -DLLVM_DIR="$INSTALL_ROOT"/lib/cmake/llvm \
  -DMLIR_DIR="$INSTALL_ROOT"/lib/cmake/mlir \
  -DClang_DIR="$INSTALL_ROOT"/lib/cmake/clang \
  -DLLVM_EXTERNAL_LIT="$LIT_PATH" \
  -DGTEST_ROOT="$INSTALL_ROOT" \
  -DLLZK_BUILD_DEVTOOLS=ON
