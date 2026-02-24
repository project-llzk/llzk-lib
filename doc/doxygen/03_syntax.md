# LLZK Language Specification {#syntax}

\tableofcontents

## Syntax

The root `module` in LLZK IR must have the `llzk.lang` attribute with an optional string that is typically used to indicate the source language. The root `module` can contain any number of `struct.def`, `function.def`, or other `module` ops. The `struct.def` op is the foundation of LLZK IR and is used to describe each component in a circuit. It can contain any number of data members, a `compute()` function that holds the witness generation code, and a `constrain()` function that holds that constraint generation code. No other functions may appear within a `struct.def`.

Here is a simple example of LLZK IR translated from the circomlib [and gate](\ref circomlib-and-gate):

```LLZK IR
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

<!--
## Types

- `i1`: subtype of Felt in [0,1]
- `index`: machine integer
- `felt.type`: finite field element
- `struct.type` (component): Aggregate type with named heterogeneous elements. Generally correlates to components/functions in the source language. Constituent elements may be local variables, subcomponents, and/or called functions.
- `array<E>`: elements can be any type, including other array type for multi-dimensional arrays. We may need this type to have a built-in field named `len` that returns the length of the array.
- `const<T>`: **modifier** on types to denote it’s a compile-time constant. Semantic analysis can infer `const` based on usage of literal values, etc. but it can also be specified in the IR in which case the semantic analysis must ensure it’s correct or give an error. The semantics of several syntax nodes require a `const` value, such as the `i` in `GetWeight<i>.compute()`. Global function return type can be `const` and that would allow such a function to be used in these locations.

## Special Constructs

- `nondetFelt()`: can be used as the parameter of a `constrain()` function when the expression from the source language can be elided because it cannot be used as part of a constraint. For example, expressions containing bitwise operators cannot be part of a constraint.
- A special element can be added to a `struct` to store the return value of a component (i.e., `synthetic_return` in the examples above).

## Semantic Rules

- `emit` must only appear in `constrain()` functions and `return` only in global functions.
- `constrain()` only calls `constrain()` and `compute()` only calls `compute()`. Either can call arbitrary functions but not each other.
- Function parameters are pass-by-value.
- For references like `a.b`, a field named `b` must be present in the component `a`. For references like `c`, if a field named `c` is present in the current component the reference refers to that field instance, otherwise it refers to a local named `c` within that function, with type inferred from the RHS of the expression where `c` is defined.
- Global constants cannot be modified/assigned.
-->

## Translation Guidelines {#translation-guidelines}

- The frontend translation for each source language to LLZK IR should be as simple as possible since this will be repeated effort for each source language. To expand support of frontend languages, we welcome proposals of new high-level syntax along with a translation of that syntax to existing LLZK syntax.
- To promote reusable infrastructure, transformations or optimizations should be performed on the LLZK IR rather than the source language, whenever possible. We welcome PRs to LLZK-lib for reusable passes.
- Loops can be represented with either `scf.for` or `scf.while` and the optional `llzk.loopbounds` attribute can be added to specify known iteration information.
- Frontend tranlations should attach accurate source line information to operations via the `Location` whenever possible.
- Only the outermost module should have the `llzk.lang` attribute (because the presence of that attribute is used to determine the “root” symbol table for symbol resolution).
- All inner modules must be named because their names are used to build the fully-qualified path names for symbol references.
- All references to functions and types must use fully-qualified paths.

[circomlib-and-gate]: https://github.com/iden3/circomlib/blob/master/circuits/gates.circom#L29-L35
