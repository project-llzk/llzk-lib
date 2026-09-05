# LLZK Language Specification {#syntax}

\tableofcontents

## Syntax

The root `module` in LLZK IR must have the `llzk.lang` attribute with an optional string that is typically used to indicate the source language. The root `module` can contain any number of `struct.def`, `function.def`, or other `module` ops. The `struct.def` op is the foundation of LLZK IR and is used to describe each component in a circuit. It can contain any number of data members, a `compute()` function that holds the witness generation code, and a `constrain()` function that holds the constraint generation code. No other functions may appear within a `struct.def`.

Here is a simple example of LLZK IR translated from the circomlib [and gate][circomlib-and-gate]:

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
- `array.type<N x E>`: Aggregate type with indexed [pseudo-homogeneous](\ref pseudo-homogeneous) elements. Element type cannot be another array type, instead multi-dimensional arrays are specified with a comma-separated list of dimension sizes. Each dimension size can be specified as an integer literal, a flat symbol reference to an index-typed `poly.param` or `poly.expr` binding, a fully-qualified reference to an index-typed constant global, or a single-result [affine_map](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures) (used when creating arrays within a loop where the dimension size depends on the loop iteration variable).
- `struct.type<@Name<[...]>>`: Aggregate type whose named heterogeneous elements are declared within `struct.def @Name`. It generally correlates to a component or function in the source language.
  For a definition nested in a `poly.template`, an optional instantiation list supplies one argument per `poly.param`, in declaration order. For a definition with no parameters, the list may be omitted or explicitly written as `[]`.
  Each argument is one of the following:
  - an integer literal;
  - a felt constant such as `#felt<const 35>`;
  - a flat symbol reference to a `poly.param` or `poly.expr` binding or a fully-qualified reference
    to a constant global;
  - a valid LLZK type used to instantiate a `poly.tvar<@N>` (see below); or
  - a single-result [affine_map](https://mlir.llvm.org/docs/Dialects/Affine/#polyhedral-structures) for an integer-like argument that depends on a loop iteration variable.

  A type argument may contain any valid LLZK type. The verifier recursively resolves references in
  `struct.type` and `array.type` arguments and resolves a `poly.tvar` parameter reference at the
  `struct.type` use site. Record member types within a `pod.type` are not traversed.

  The following schematic arguments assume `poly.template @T` declares
  `poly.param @P : !felt.type<"bn128">` and contains `struct.def @S`. They select the same value and
  field:

  ```llzk
  // The fieldless constant is accepted as a bn128 value for @P.
  !struct.type<@T::@S<[#felt<const 35>]>>
  // The field can also be written explicitly.
  !struct.type<@T::@S<[#felt<const 35 : !felt.type<"bn128">>]>>
  ```
- `pod.type<..>`: Plain Old Data aggregate type with named heterogeneous elements. Unlike `struct.type`, there is no associated named declaration, the type itself specifies all constituent element types. It can be used more freely than `struct.type` since it has fewer restrictions on modifications.
- `poly.tvar<@N>`: Placeholder type variable whose name refers to `poly.param @N` in an enclosing
  `poly.template` and may be instantiated with different types. The parameter may be unrestricted;
  declaring it as `poly.param @N : !poly.tvar<@N>` restricts it to type arguments and enables type
  inference for that parameter.
- `string.type`: Sequence of characters.

### Pseudo-homogeneous arrays {#pseudo-homogeneous}

LLZK supports arrays where the element type is not truly homogeneous, specifically when a templated `struct.type` is used with an `affine_map` parameter. For example, the type `!array.type<10 x !struct.type<@X<[affine_map<(i)[] -> (i*5)>]>>>` contains instances of the struct `@X` instantiated with different parameter values per `affine_map<(i)[] -> (i*5)>`. Use of this type can be seen in [circom_example_2.llzk](https://github.com/project-llzk/llzk-lib/blob/main/test/FrontendLang/Circom/circom_example_2.llzk). If the circuit is ultimately instantiated and flattened, the array will have to be split into scalar values since the instantiated struct type of each element is different.

## Semantic Rules

- Each `array.new` operation creates a fresh mutable array allocation. Two identical `array.new` operations are not interchangeable when either result may be read or written. The same is true for `pod.new`.
- Felt-valued template arguments on `struct.type`, templated free-function calls, and `verif.include` may be integer literals, felt constants, or symbols.
  A fieldless felt restriction accepts any felt field. For a fielded restriction, a fieldless felt constant or integer is accepted as a value in the required field, while an explicitly fielded constant must use that field. A symbol is accepted only when the referenced binding or global has the required field type; a fieldless or absent type is rejected.
  For a felt-valued parameter inferred independently from multiple positions, all known fields and concrete values must agree.
- A symbolic template argument must refer to a `poly.param` or `poly.expr` binding by its flat name
  or to a constant global by its fully-qualified name; unresolved symbols, mutable globals, and
  symbols that resolve to another operation kind are rejected.
  For an operation nested in a `verif.contract` whose target is nested in a `poly.template`, flat
  binding names resolve in the target's template; otherwise they resolve in the nearest enclosing
  template.
  A binding without a declared type restriction may remain deferred for an index, integer, or fieldless felt parameter. It cannot satisfy a fielded felt or `poly.tvar` restriction, and an array dimension requires an index-typed binding at that type use.
- A direct `function.call` or `verif.include` argument for an index or integer restriction must be an integer, not an affine map. A single-result affine map remains valid as a `struct.type` argument with such a restriction.
- The `?` wildcard in a `function.call` or `verif.include` template argument is valid only for a `poly.tvar` restriction and leaves its concrete type for later inference from the target body. An array dimension may independently use `?` as a dynamic size.
- A type argument must be a valid LLZK type. A `function.call` or `verif.include` verifier
  recursively resolves references in `struct.type` and `array.type` arguments and resolves a
  `poly.tvar` parameter reference at the call or inclusion site. A `poly.tvar` is accepted once its
  parameter reference resolves, even when its concrete type remains deferred. Record member types
  within a `pod.type` are not traversed.
- A `function.def` argument may have `function.arg_name = "..."` to preserve the source-level argument name independently from the SSA name printed by MLIR. The value must be a non-empty, untyped string attribute; typed string attributes such as `"x" : i1` are rejected. Attached argument names must be unique within the function. Argument-splitting transforms derive names for generated arguments, such as `input[0]` for array elements or `self.member` for struct members.
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
- References to function and type definitions must use fully-qualified paths. Template arguments
  use flat names for `poly.param` and `poly.expr` bindings and fully-qualified names for constant
  globals.

[circomlib-and-gate]: https://github.com/iden3/circomlib/blob/master/circuits/gates.circom#L29-L35
