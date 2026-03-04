## v1.1.1 - 2026-03-04
### Fixed
- properly update version number

## v1.1.0 - 2026-03-03
### Added
- CAPI:
  - Added function that generates the PCL lisp representation of a circuit.
### Fixed
- Release workflow
## v1.0.0 - 2026-03-02
### Added
- LLZK IR:
  - core dialects
  - common helper classes and functions for working with LLZK IR
  - unit tests and lit tests
  - documentation pages
- Passes:
  - Analysis:
    - "`llzk-print-call-graph` - Print the call graph"
    - "`llzk-print-call-graph-sccs` - Print Strongly Connected Components in the call graph"
    - "`llzk-print-constraint-dependency-graphs` - Print constraint dependency graph for all LLZK structs"
    - "`llzk-print-interval-analysis` - Print interval analysis results for all LLZK structs"
    - "`llzk-print-symbol-def-tree` - Print symbol definition tree"
    - "`llzk-print-symbol-use-graph` - Print symbol use graph"
    - "`llzk-print-predecessors` - Print the predecessors of all operations"
  - Transformation:
    - "`llzk-array-to-scalar` - Replace arrays with scalar values"
    - "`llzk-inline-includes` - Replace all IncludeOp with contents of the referenced file"
    - "`llzk-drop-empty-params` - Remove empty struct parameter lists"
    - "`llzk-flatten` - Flatten structs (i.e. replace parameterized with instantiated) and unroll loops"
    - "`llzk-duplicate-read-write-elim` - Remove redundant reads and writes"
    - "`llzk-duplicate-op-elim` - Remove redundant arith operations"
    - "`llzk-unused-declaration-elim` - Remove unused member and struct declarations"
    - "`llzk-poly-lowering-pass` - Lowers the degree of all polynomial equations to a specified maximum"
    - "`llzk-inline-structs` - Inline nested structs (i.e., subcomponents)"
    - "`llzk-compute-constrain-to-product` - Replace @compute/@constrain with @product function"
    - "`llzk-fuse-product-loops` - Fuse matching witness/constraint loops in a @product function"
  - Conversion:
    - "`llzk-to-pcl` - Rewrite constraints to be compatible with PCL constraints"
    - "`llzk-r1cs-lowering` - Rewrite constraints to be compatible with R1CS constraints"
    - "`convert-llzk-to-zklean` - Convert LLZK structs and constraints to ZKLean IR"
    - "`convert-zklean-to-llzk` - Convert ZKLean IR back to LLZK structs and constraints"
    - "`zklean-pretty-print` - Pretty-print zkLean dialect IR as Lean code"
  - Validation:
    - "`llzk-validate-member-writes` - Detect multiple and missing writes to the same member of a component"
- llzk-tblgen tool that generates C API for llzk library and linking unit tests for it
- CAPI for LLZK IR and passes
- Backends:
  - r1cs
  - zklean
  - pcl
