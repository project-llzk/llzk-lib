# Architecture {#overview}

\tableofcontents

## Project Overview

The LLZK project consists of three main components:

1. **Frontends**, which translate a source ZK language to LLZK.
2. **Passes**, which analyze or transform the IR.
3. **Backends**, which process LLZK into another destination format (e.g., R1CS constraints) or analyze the IR to identify bugs or verify properties of the source language.

The general workflow of using LLZK is therefore as follows:
1. Translate the source language into LLZK using [a frontend tool](\ref frontends).
2. Use the [llzk-opt tool](\ref llzk-opt) to perform any transformations using LLZK's [pass infrastructure](\ref pass-overview).
3. Provide the transformed IR to a [backend](\ref backends) for further analysis or use.

### Frontends {#frontends}

Frontends are currently not contained within the LLZK repository, but are rather
maintained in separate repositories, using LLZK-lib as a dependency.

Veridise currently maintains the following frontends:
- [haloumi](https://github.com/project-llzk/haloumi)
<!-- TODO: Update this link to a doxygen site at some point. -->

For information on how to create a new frontend, please refer to the \ref translation-guidelines.

### Passes {#pass-overview}

LLZK provides three types of passes:
1. *Analysis* passes, which compute useful information about the IR used to implement other passes or backends.
2. *Transformation* passes, which restructure or optimize the IR.
3. *Validation* passes, which ensure the IR has certain required properties.

User documentation about how to use these passes is provided in \ref tools.

Developer documentation can be found:
- In the Analysis directories:
    - General, multi-dialect analyses: \ref include/llzk/Analysis, \ref lib/Analysis
- In the Transforms directories:
    - General, multi-dialect transforms: \ref include/llzk/Transforms, \ref lib/Transforms
    - `array` transforms: \ref include/llzk/Dialect/Array/Transforms, \ref lib/Dialect/Array/Transforms
    - `polymorphic` transforms: \ref include/llzk/Dialect/Polymorphic/Transforms, \ref lib/Dialect/Polymorphic/Transforms
- In the Validators directories
    - General, multi-dialect validators: \ref include/llzk/Validators, \ref lib/Validators

### Backends {#backends}

The LLZK project currently maintains the following backends:
- [R1CS](\ref r1cs)

Veridise also maintains a [Picus Contraint Language backend](https://github.com/Veridise/pcl-mlir) that
allows LLZK to be lowered to PCL for use with the [Picus][picus-v2] verifier.


<div class="section_buttons">
| Previous          |                              Next |
|:------------------|----------------------------------:|
| \ref mainpage | \ref setup |
</div>

[picus-v2]: https://docs.veridise.com/picus-v2/
[zk-vanguard]: https://docs.veridise.com/zkvanguard/
