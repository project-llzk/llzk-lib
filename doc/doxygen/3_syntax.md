# LLZK Language Specification {#syntax}

<!--
TODO: Update this specification (https://github.com/project-llzk/llzk-lib/issues/262)
Based on: https://www.notion.so/veridise/ZK-IR-Design-5d4e6b675c9142e9b1583bdca3c8c8a6
-->

\todo This documentation is out of date and will be updated shortly.

\tableofcontents

## Syntax

```EBNF

program = program_header, { include | global_const | global_func | component } ;
program_header = "#dialect", identifier ;
include = "#include", ? path ? ;
global_const = identifier, ":=", const_expr, ";" , [ annotation ] ;
global_func = "function.def", identifier, parameters, "->", type, body ;
component = "struct", identifier, [ struct_params ],
						"{", fields, funcs, "}", [ annotation ] ;
struct_params = "<", [ identifier, { "," , identifier }], ">" ;
fields = { identifier, ":", type, "," , [ annotation ] } ;
funcs = ( compute , constrain ) | ( constrain, compute ) ;
compute = "function.def compute", parameters, body ;
constrain = "function.def constrain", parameters, body ;
parameters = "(", [ param, { "," , param }], ")" ;
param = identifier, ":", type ;
body = "{", { statement, ";", [ annotation ] }, "}" ;
statement = emit | return | while | if | assign | expr_call ;
emit = "emit", ( expr_contains | ( expr, "=", expr ) ) ;
return = "return", expr ;
while = "while", expr , body ;
if = "if", expr, body, [ "else", body ] ;
assign = lvalue, ":=", expr ;
const_expr = expr ; (* semantics require const type *)
expr = literal | lvalue | expr_array | expr_call | expr_ref |
       expr_contains | expr_binop | expr_monop | expr_special ;
expr_array = "[", list_of_expr, "]" ;
expr_call = [ expr_call_base ], expr_call_target ;
expr_call_base = lvalue, [ "<", const_expr, ">" ], "." ;
expr_call_target = identifier, "(", list_of_expr, ")" ;
expr_ref = lvalue, ".", identifier ;
expr_contains = lvalue, "in", lvalue ;
expr_binop = expr, binop, expr ;
list_of_expr = [ expr, { "," , expr }] ;
binop = binop_arith | binop_bit | binop_compare | binop_logic ;
binop_arith = "+" | "-" | "*" | "/" | "%" ;
binop_bit = "&" | "|" | "^" | "<<" | ">>" ;
binop_compare = ">" | "<" | ">=" | "<=" | "==" | "!=" ;
binop_logic = "&&" | "||" ;
expr_monop = ( "(", expr, ")" ) | ( monop, expr ) ;
monop = monop_arith | monop_bit | monop_logic ;
monop_arith = "-" ;
monop_bit = "~" ;
monop_logic = "!" ;
expr_special = "nondetFelt()";
lvalue = identifier | lvalue_array ;
lvalue_array = lvalue, "[", lvalue, "]" ;
type = { type_modifier }, type_base ;
type_modifier = "pub" | "const" ;
type_base = identifier | ( type, "[", const_expr, "]" ) ;
constant = literal | identifier ;
identifier = letter, { letter | digit } ;  (* and symbols if needed *)
annotation = "#", ? anything except newline ? ;
letter = ? all capital and lowercase characters ? ;
literal = ( digit, { digit } ) | ( "0x", hexit, { hexit } ) ;
hexit = digit | "A" | "B" | "C" | "D" | "E" | "F"
              | "a" | "b" | "c" | "d" | "e" | "f" ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;

```

## Types

- `bool`: subtype of Felt in [0,1]
- `int`: machine integer
- `Felt`: finite field element
- `struct` (component): Aggregate type with named heterogeneous elements. Generally correlates to components/functions in the source language. Constituent elements may be local variables, subcomponents, and/or called functions.
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

## Translation Guidelines {#translation-guidelines}

- The modifier `pub` can be added before the type on `compute()` and `constrain()` parameters to denote which elements of the domain are public (default is private). Likewise, it can be used on struct fields to denote which fields are part of the co-domain (i.e., public outputs).
- The frontend translation for each source language to ZK IR should be as simple as possible since this will be repeated effort for each source language. Any transformations or optimizations on the ZK IR should be done in a shared module run on the internal representation after translation.
- Constant integer parameters on a `struct` can be used to avoid creating multiple versions of that `struct` in the IR. A later pass can flatten the IR if needed by the client analysis. This is even where multiple dialects of the IR could be used, with a flattened dialect disallowing these parameters.
- If loop bounds are known, `scf.for` should be used to make loop bounds explicit. However, `scf.while` is available to handle the general case if that information is not available but this should not be used in the `constrain()` function.
- Global functions (i.e., user-defined or helper functions, located outside of `struct` definitions) are pure. There is no global state and parameters are pass-by-value (i.e., a copy is created) so there is nothing external they can modify.
- Source line information should be handled via MLIR so frontend components must provide that information when building the MLIR AST.
- Only the outermost module should have the `veridise.lang = "llzk"` attribute (because the presence of that attribute is used to determine the “root” symbol table for symbol resolution).
- All inner modules must be named because their names are used to build the fully-qualified path names for symbol references.
- All references to functions and types must use fully-qualified paths.
