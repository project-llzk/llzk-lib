

The development of zkLean dialects added to LLZK is done in the following files around the LLZK
repo:

```
.
├── CMakeLists.txt
├── build_deps.sh
├── include
│   ├── CMakeLists.txt
│   └── llzk
│       ├── Conversations
│       │   ├── CMakeLists.txt
│       │   ├── Passes.h
│       │   └── Passes.td
│       ├── Dialect
│       │   ├── ZKBuilder
│       │   │   └── IR
│       │   │       ├── CMakeLists.txt
│       │   │       ├── ZKBuilderDialect.h
│       │   │       ├── ZKBuilderDialect.td
│       │   │       ├── ZKBuilderOps.h
│       │   │       ├── ZKBuilderOps.td
│       │   │       ├── ZKBuilderTypes.h
│       │   │       └── ZKBuilderTypes.td
│       │   ├── ZKExpr
│       │   │    └── IR
│       │   │        ├── CMakeLists.txt
│       │   │        ├── ZKExprDialect.h
│       │   │        ├── ZKExprDialect.td
│       │   │        ├── ZKExprOps.h
│       │   │        ├── ZKExprOps.td
│       │   │        ├── ZKExprTypes.h
│       │   │        └── ZKExprTypes.td
│       │   └── ZKLeanLean
│       │       └── IR
│       │           ├── CMakeLists.txt
│       │           ├── ZKLeanLeanDialect.h
│       │           ├── ZKLeanLeanDialect.td
│       │           ├── ZKLeanLeanOps.h
│       │           ├── ZKLeanLeanOps.td
│       │           ├── ZKLeanLeanTypes.h
│       │           └── ZKLeanLeanTypes.td
│       └── Transforms
│           └── ZKLeanPasses.h
├── lib
│   ├── CMakeLists.txt
│   ├── Conversations
│   │   ├── CMakeLists.txt
│   │   ├── LLZKToZKLean.cpp
│   │   └── ZKLeanToLLZK.cpp
│   ├── Dialect
│   │   ├── ZKBuilder
│   │   │   └── IR
│   │   │       ├── CMakeLists.txt
│   │   │       ├── ZKBuilderDialect.cpp
│   │   │       └── ZKBuilderOps.cpp
│   │   ├── ZKExpr
│   │   │    └── IR
│   │   │       ├── CMakeLists.txt
│   │   │       ├── ZKExprDialect.cpp
│   │   │       └── ZKExprOps.cpp
│   │   └── ZKLeanLean
│   │       ├── CMakeLists.txt
│   │       └── IR
│   │           ├── ZKLeanLeanDialect.cpp
│   │           └── ZKLeanLeanOps.cpp
│   ├── InitDialects.cpp
│   └── Transforms
│       ├── CMakeLists.txt
│       └── PrettyPrintZKLean.cpp
├── test
│   ├── Conversations
│   │   ├── circom_isZero.llzk
│   │   └── zklean_isZero.llzk
│   └── Dialect
│       ├── ZKBuilder
│       │   └── zkbuilder_syntax.mlir
│       └── ZKExpr
│           └── zkexpr_syntax.mlir
└── tools
    └── llzk-opt
        ├── CMakeLists.txt
        └── llzk-opt.cpp
```


# Summary:

LLZK \<-\> ZKLean IR -\> ZKLean pipeline implemented and tested on LLZK circom isZero test file.

* Registers ZKLeanLean, ZKExpr and ZKBuilder dialects with LLZK
* Registers `--zklean-pretty-print` pass with LLZK
* Implements `--convert-llzk-to-zklean `pass
* Implements `--convert-zklean-to-llzk `pass

## LLZK -\> ZKLean:

* Creates new @ZKLean module
* Converts struct.\* to ZKLeanLean.\* (defs and field accessors)
* Converts felt.\* and constrain.\* ops to ZKExpr.\* and ZKBuilder.\* ops, respectively
* Converts bool.cmp and cast.tofelt to ZKLeanLean.call for Lean-side function resolution
* felt.constants remain in the ZKLean IR as inputs to ZKExpr.Literals
* Strips out struct types in function arguments

## ZKLean -\> LLZK:

* Converts ZKExpr.\* and ZKBuilder.\* ops to felt.\* and constrain.\* ops, respectively
* Converts ZKLeanLean.\* to struct.\*, with an empty @compute function to satisfy the LLZK condition that struct needs both @compute and @constrain functions.
* ZKExpr.Witnessable.witness converted to new arguments of felt type to the LLZK function
* Converts ZKLeanLean.call back to bool.cmp and cast.tofelt when applicable

## Misc:

* ZKExpr.Literal now takes as inputs LLZK felts

# Build Instructions:

## Prerequisites:

LLZK requires:

* CMake 3.18 or newer
* Ninja
* Z3 (NOTE: included bash script does **not** build LLVM with Z3 but is not necessary to run conversion / zklean pretty printing passes, see `build_deps.sh` to enable Z3 build)
* Clang 16 or higher (use the same compiler for both LLVM and LLZK repos)

To run tests, you also need:

* Python 3
* llvm-lit
* gtest

## Build:

While standing in the top-level LLZK directory, do

`bash build_deps.sh`

This will build LLVM 20.1.8 in third-party/ and create a build/ directory, pointing LLZK at LLVM.

**Warning:** this script will also do some strange sym linking in your home directory, see `build_deps.sh` for details.

# Usage:

While standing in top level LLZK directory:

1. To build llzk-opt (and other binaries), do:

   `cmake --build build/`
2. To convert LLZK circom isZero circuit into ZKLean IR, do:

   `./build/bin/llzk-opt --convert-llzk-to-zklean test/Conversions/circom_isZero.llzk`
3. To convert ZKLean IR isZero circuit constraints into LLZK, do:

   `./build/bin/llzk-opt --convert-zklean-to-llzk test/Conversions/zklean_isZero.llzk`
4. To pretty print ZKLean IR into ZKLean, do:

   `./build/bin/llzk-opt --zklean-pretty-print test/Conversions/zklean_isZero.llzk`
