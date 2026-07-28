The zklean IR files live under `backends/zklean/` and mirrors the layout of
other backends (such as `r1cs`). The key files are:

```
backends
└── zklean
    ├── include
    │   └── zklean
    │       ├── Conversions
    │       │   ├── CMakeLists.txt
    │       │   ├── Passes.h
    │       │   └── Passes.td
    │       ├── Dialect
    │       │   ├── ZKBuilder/IR/{CMakeLists.txt, *.td, *.h}
    │       │   ├── ZKExpr/IR/{CMakeLists.txt, *.td, *.h}
    │       │   └── ZKLeanLean/IR/{CMakeLists.txt, *.td, *.h}
    │       ├── Transforms
    │       │   ├── CMakeLists.txt
    │       │   └── ZKLeanPasses.{h,td}
    │       └── DialectRegistration.h
    └── lib
        ├── Conversions
        │   ├── CMakeLists.txt
        │   ├── LLZKToZKLean.cpp
        │   └── ZKLeanToLLZK.cpp
        ├── Dialect
        │   ├── CMakeLists.txt
        │   ├── ZKBuilder/IR/ZKBuilderDialect.cpp
        │   ├── ZKExpr/IR/ZKExprDialect.cpp
        │   └── ZKLeanLean/IR/ZKLeanLeanDialect.cpp
        ├── Transforms
        │   ├── CMakeLists.txt
        │   └── PrettyPrintZKLean.cpp
        └── DialectRegistration.cpp
```


# Summary:

LLZK \<-\> ZKLean IR -\> ZKLean pipeline implemented and tested on LLZK circom isZero test file.

* Registers ZKLeanLean, ZKExpr and ZKBuilder dialects with LLZK
* Registers `--zklean-pretty-print` pass with LLZK
* Implements `--convert-llzk-to-zklean `pass
* Implements `--convert-zklean-to-llzk `pass

## LLZK -\> ZKLean:

* Replaces the input module with @ZKLean
* Converts struct.\* to ZKLeanLean.\* (defs and member accessors)
* Converts felt.\* and constrain.\* ops to ZKExpr.\* and ZKBuilder.\* ops, respectively
* Converts bool.cmp and cast.tofelt to ZKLeanLean.call for Lean-side function resolution
* felt.constants remain in the ZKLean IR as inputs to ZKExpr.Literals
* Strips out struct types in function arguments

## ZKLean -\> LLZK:

* Converts ZKExpr.\* and ZKBuilder.\* ops to felt.\* and constrain.\* ops, respectively
* Converts ZKLeanLean.\* to struct.\*, with an empty @compute function to satisfy the LLZK condition that struct needs both @compute and @constrain functions.
* ZKBuilder.AllocWitness converted to new arguments of felt type to the LLZK function
* Converts ZKLeanLean.call back to bool.cmp and cast.tofelt when applicable

## Misc:

* ZKExpr.Literal now takes as inputs LLZK felts

# Build Instructions:

Please see the LLZK's [Setup and Development Page](https://project-llzk.github.io/llzk-lib/main/setup.html). Follow the Manual Build Setup and Development Workflow.

This will build LLVM 20.1.8 in third-party/ and create a build/ directory,
pointing LLZK at LLVM. (NOTE: Z3 is included but not necessary to run
conversion / zklean pretty printing passes.)

# Usage:

While standing in top level LLZK directory:

1. To build llzk-opt (and other binaries), do:

   `cmake --build build/`
2. To generate dialect docs, do the following and the generated markdown files are available in `build/doc/mlir/dialect/*.md`:

   `cmake --build build/ --target doc`
3. To convert LLZK circom isZero circuit into ZKLean IR, do:

   `./build/bin/llzk-opt --convert-llzk-to-zklean test/Conversions/circom_isZero.llzk`
4. To convert ZKLean IR isZero circuit constraints into LLZK, do:

   `./build/bin/llzk-opt --convert-zklean-to-llzk test/Conversions/zklean_isZero.llzk`
5. To pretty print ZKLean IR into ZKLean, do:

   `./build/bin/llzk-opt --zklean-pretty-print test/Conversions/zklean_isZero.llzk`


# Test

To test the conversions between llzk to zklean on the simple test files:
```bash
llzk-opt --convert-llzk-to-zklean test/Conversions/circom_isZero.llzk 2>&1 | FileCheck --enable-var-scope test/Conversions/circom_isZero.llzk
llzk-opt --convert-zklean-to-llzk test/Conversions/zklean_isZero.llzk 2>&1 | FileCheck --enable-var-scope test/Conversions/zklean_isZero.llzk
```
