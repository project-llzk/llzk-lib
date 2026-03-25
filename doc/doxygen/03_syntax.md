# LLZK Language Specification {#syntax}

\tableofcontents

## Syntax

The root `module` in LLZK IR must have the `llzk.lang` attribute with an optional string that is typically used to indicate the source language. The root `module` can contain any number of `struct.def`, `function.def`, or other `module` ops. The `struct.def` op is the foundation of LLZK IR and is used to describe each component in a circuit. It can contain any number of data members, a `compute()` function that holds the witness generation code, and a `constrain()` function that holds the constraint generation code. No other functions may appear within a `struct.def`.

Here is a simple example of LLZK IR translated from the circomlib [and gate](\ref circomlib-and-gate):

```mlir
module attributes {llzk.lang = "circom"} {
  struct.def @AND {
    struct.member @out : !felt.type {llzk.pub}
    function.def @compute(%a: !felt.type, %b: !felt.type) -> !struct.type<@AND> {
      %self = struct.new : !struct.type<@AND>
      %0 = felt.mul %a, %b : !felt.type, !felt.type
      struct.writem %self[@out] = %0 : !struct.type<@AND>, !felt.type
      function.return %self : !struct.type<@AND>
    }
    function.def @constrain(%self: !struct.type<@AND>, %a: !felt.type, %b: !felt.type) {
      %0 = struct.readm %self[@out] : !struct.type<@AND>, !felt.type
      %1 = felt.mul %a, %b : !felt.type, !felt.type
      constrain.eq %0, %1 : !felt.type, !felt.type
      function.return
    }
  }
}
```

## Types

- `i1`: (MLIR builtin) Boolean value [0,1].
- `index`: (MLIR builtin) Machine integer.
- `felt.type`: Finite field element.
- `array.type<N x E>`: Aggregate type with indexed [pseudo-homogeneous](\ref pseudo-homogeneous) elements. Element type cannot be another array type, instead multi-dimensional arrays are specified with a comma-separated list of dimension sizes. Each dimension size can be specified as an integer literal, a symbol (referring to a template parameter within a templated `struct.def`), or an [affine_map](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures) (used when creating arrays within a loop where the dimension size depends on the loop iteration variable).
- `struct.type<[..]>`: Aggregate type with named heterogeneous elements corresponding to a `struct.def`. Generally correlates to components/functions in the source language. Constituent elements may be local variables, subcomponents, and/or called functions. Optionally includes a list of parameters to instantiate a templated `struct.def` where each parameter can be an integer literal, a symbol (referring to a template parameter within a templated `struct.def`), a type used to instantiate a `poly.tvar<@N>` (see below), or an [affine_map](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures) (used when the parameter of a templated `struct.type` depends on a loop iteration variable).
- `pod.type<..>`: Plain Old Data aggregate type with named heterogeneous elements. Unlike `struct.type`, there is no associated named declaration, the type itself specifies all constituent element types. It can be used more freely than `struct.type` since it has fewer restrictions on modifications.
- `poly.tvar<@N>`: Placeholder type variable within a templated `struct.def` that may be instantiated with different types.
- `string.type`: Sequence of characters.

### Pseudo-homogeneous arrays {#pseudo-homogeneous}

LLZK supports arrays where the element type is not truly homogeneous, specifically when a templated `struct.type` is used with an `affine_map` parameter. For example, the type `!array.type<10 x !struct.type<@X<[affine_map<(i)[] -> (i*5)>]>>>` contains instances of the struct `@X` instantiated with different parameter values per `affine_map<(i)[] -> (i*5)>`. Use of this type can be seen in [circom_example_2.llzk](\ref test/FrontendLang/Circom/circom_example_2.llzk). If the circuit is ultimately instantiated and flattened, the array will have to be split into scalar values since the instantiated struct type of each element is different.

## Semantic Rules

- Ops marked with the `WitnessGen` trait can only be used in functions with the `allow_witness` attribute (`compute()` within `struct.def` has this by default). Similarly, ops marked with the `ConstraintGen` trait can only be used in functions with the `allow_constraint` attribute (`constrain()` within `struct.def` has this by default).
- Functions with the `allow_witness` attribute can only call other functions marked with `allow_witness`. Likewise for `allow_constraint`.
- Ops marked with the `NotFieldNative` trait can only be used in functions with the `allow_non_native_field_ops` attribute. Some of these ops have known transformations to field-native operations but others do not. It is up to backend users to determine how to handle such ops appearing in `constrain()` functions (one possibility being replacing these ops with `llzk.nondet`)

## Translation Guidelines {#translation-guidelines}

- The frontend translation for each source language to LLZK IR should be as simple as possible since this will be repeated effort for each source language. To expand support of frontend languages, we welcome proposals of new high-level syntax along with a translation of that syntax to existing LLZK syntax.
- To promote reusable infrastructure, transformations or optimizations should be performed on the LLZK IR rather than the source language, whenever possible. We welcome PRs to LLZK-lib for reusable passes.
- Loops can be represented with either `scf.for` or `scf.while` and the optional `llzk.loopbounds` attribute can be added to specify known iteration information.
- Frontend translations should attach accurate source line information to operations via the `Location` whenever possible.
- Only the outermost module should have the `llzk.lang` attribute (because the presence of that attribute is used to determine the “root” symbol table for symbol resolution).
- All inner modules must be named because their names are used to build the fully-qualified path names for symbol references.
- All references to functions and types must use fully-qualified paths.

[circomlib-and-gate]: https://github.com/iden3/circomlib/blob/master/circuits/gates.circom#L29-L35
