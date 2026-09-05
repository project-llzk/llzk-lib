//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_POLYMORPHIC_OPS
#define LLZK_POLYMORPHIC_OPS

include "llzk/Dialect/Polymorphic/IR/Dialect.td"
include "llzk/Dialect/Polymorphic/IR/Types.td"
include "llzk/Dialect/Polymorphic/IR/OpInterfaces.td"
include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class PolymorphicDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<PolymorphicDialect, mnemonic, traits>;

def LLZK_TemplateOp
    : PolymorphicDialectOp<"template", [HasParent<"::mlir::ModuleOp">, Symbol,
                                        LLZKSymbolTable, IsolatedFromAbove,
                                        NoRegionArguments, NoTerminator,
                                        SingleBlock]> {
  let summary = "defines polymorphic functions or structs";
  let description = [{
    The `poly.template` allows defining polymorphic templated functions and structs.
    The body contains the definitions of the template's parameters along with the
    function and/or struct definitions that utilize those parameters in their bodies.

    Example:
    ```llzk
    poly.template @TemplateName {
      poly.param @N
      poly.param @T

      function.def @f(%inp: !array.type<8,5,@N x !felt.type>) -> !array.type<@N x !felt.type> {
        // function header and body can use parameters @N and @T
      }

      struct.def @StructName {
        function.def @compute() -> !struct.type<@TemplateName::@StructName> {
          ...
        }
        function.def @constrain(%self: !struct.type<@TemplateName::@StructName>) {
          ...
        }
      }
    }
    ```

    The order of `poly.param` definitions in the template body determines the order that
    template parameters must be listed in the parameter list of a `struct.type` referring
    to a struct nested within the template. In the example above, the type of `@StructName`
    is `!struct.type<@TemplateName::@StructName<[@N, @T]>>`.

    Transform passes may attach the discardable generic attribute `poly.name_pattern` to a partial
    template with `P` current `poly.param` operations. Its value is an `ArrayAttr` of `P + 1`
    literal `StringAttr` chunks; gap `i`, between chunks `i` and `i + 1`, corresponds to current
    parameter `i`. The chunks are carried as structured metadata and are never recovered by
    interpreting `sym_name`; without the attribute, every byte of the current `sym_name` is opaque
    source or legacy text. When a transform closes the final gap, it removes the attribute rather
    than retaining a terminal one-chunk state.
  }];

  let hasVerifier = 1;

  let arguments = (ins SymbolNameAttr:$sym_name);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let assemblyFormat = [{ $sym_name $bodyRegion attr-dict }];

  let extraClassDeclaration = [{
    /// Return ops of type `OpT` within the body region.
    /// Ops are returned in the order they are defined in the IR.
    template <TemplateSymbolBindingOp OpT>
    inline ::llvm::iterator_range<::mlir::Region::op_iterator<OpT>> getConstOps() {
      return getBodyRegion().getOps<OpT>();
    }

    /// Return `true` if there are ops of type `OpT` within the body region.
    template <TemplateSymbolBindingOp OpT>
    inline bool hasConstOps() {
      return !getConstOps<OpT>().empty();
    }

    /// Return the number of ops of type `OpT` within the body region.
    template <TemplateSymbolBindingOp OpT>
    inline size_t numConstOps() {
      return llvm::range_size(getConstOps<OpT>());
    }

    /// Return the names of all ops of type `OpT` within the body region in the order they
    /// are defined. The names are returned as `FlatSymbolRefAttr` but the more general
    /// `Attribute` type is used in the return type since that's usually what's needed.
    template <TemplateSymbolBindingOp OpT>
    ::llvm::SmallVector<::mlir::Attribute> getConstNames() {
      return ::llvm::to_vector(::llvm::map_range(getConstOps<OpT>(), [](auto p) -> ::mlir::Attribute {
        return ::mlir::FlatSymbolRefAttr::get(p.getNameAttr());
      }));
    }

    /// Return `true` if there is an op of type `OpT` with the given name within the body region.
    template <TemplateSymbolBindingOp OpT>
    inline bool hasConstNamed(::mlir::StringRef find) {
      return ::llvm::any_of(getConstOps<OpT>(), [&](OpT op) {
        return op.getName() == find;
      });
    }

    /// Return `true` if there is an op of type `OpT` with the given name within the body region.
    template <TemplateSymbolBindingOp OpT>
    inline bool hasConstNamed(::mlir::StringAttr find) {
      return hasConstNamed<OpT>(find.strref());
    }

    /// Return `true` if there is an op of type `OpT` with the given name within the body region.
    template <TemplateSymbolBindingOp OpT>
    inline bool hasConstNamed(::mlir::FlatSymbolRefAttr find) {
      return hasConstNamed<OpT>(find.getRootReference());
    }

    /// Return the op of type `OpT` with the given name within the body region if it exists, else `nullptr`.
    template <TemplateSymbolBindingOp OpT>
    inline OpT getConstNamed(::mlir::StringRef find) {
      auto range = getConstOps<OpT>();
      auto it = ::llvm::find_if(range, [&find](OpT op) { return op.getName() == find; });
      return it != range.end() ? *it : OpT{};
    }

    /// Return the op of type `OpT` with the given name within the body region if it exists, else `nullptr`.
    template <TemplateSymbolBindingOp OpT>
    inline OpT getConstNamed(::mlir::StringAttr find) {
      return getConstNamed<OpT>(find.strref());
    }

    /// Return the op of type `OpT` with the given name within the body region if it exists, else `nullptr`.
    template <TemplateSymbolBindingOp OpT>
    inline OpT getConstNamed(::mlir::FlatSymbolRefAttr find) {
      return getConstNamed<OpT>(find.getRootReference());
    }
  }];
}

def LLZK_TemplateParamOp
    : PolymorphicDialectOp<
          "param", [HasParent<"::llzk::polymorphic::TemplateOp">,
                    DeclareOpInterfaceMethods<TemplateSymbolBindingOpInterface>,
                    DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "declares a parameter of a polymorphic template";
  let description = [{
    Declares a parameter of a `poly.template` that can be used by the function and/or struct
    definitions within the template. Each parameter can have an optional type restriction.

    Example:
    ```llzk
    poly.template @TemplateName {
      poly.param @N
      poly.param @F : !felt.type
      // To restrict a parameter to accept only a type, use `!poly.tvar` with the parameter's own name.
      poly.param @T : !poly.tvar<@T>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<TypeAttrOf<ConstReadType>>:$type_opt);

  let assemblyFormat = [{ $sym_name (`:` $type_opt^)? attr-dict }];
}

def LLZK_TemplateExprOp
    : PolymorphicDialectOp<
          "expr", [HasParent<"::llzk::polymorphic::TemplateOp">,
                   DeclareOpInterfaceMethods<TemplateSymbolBindingOpInterface>,
                   DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                   IsolatedFromAbove, NoRegionArguments, SingleBlock]> {
  let summary = "declares a named expression in a polymorphic template";
  let description = [{
    Declares an expression over zero or more parameters of a `poly.template`. Its name can be used
    by `poly.read_const` and as a symbolic array dimension or template argument within the template.

    The initializer cannot reference globals, call functions, or use a `poly.expr` binding defined
    in the same template, including itself. Other operations may appear in the initializer, but
    once flattening evaluates the expression after all value dependencies and operation
    operand/result type dependencies become concrete, each `poly.read_const` must resolve to a
    concrete value, each other operation before `poly.yield` must be safe to speculate, have no
    memory effects, and fold to concrete attributes, and the value passed to `poly.yield` must
    resolve to a concrete attribute; otherwise evaluation fails. The yielded value must have an
    integral, felt, or type-variable type.

    Example:
    ```llzk
    poly.template @TemplateName {
      poly.param @N : index
      poly.expr @DoubleN {
        %n = poly.read_const @N : index
        %double = arith.addi %n, %n : index
        poly.yield %double : index
      }
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$initializerRegion);

  let assemblyFormat = [{ $sym_name $initializerRegion attr-dict }];

  let hasRegionVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the type of the `poly.yield` op in the initializer region.
    ::mlir::Type getType();
  }];
}

def LLZK_YieldOp
    : PolymorphicDialectOp<
          "yield", [HasParent<"::llzk::polymorphic::TemplateExprOp">,
                    ReturnLike, Terminator]> {
  let summary = "expr initialization yield and termination operation";
  let description = [{
    This operation yields an SSA value from a `poly.expr` initialization
    region and terminates the region.
  }];

  let arguments = (ins ConstReadType:$val);

  let assemblyFormat = [{ $val `:` type($val) attr-dict }];
}

def LLZK_ConstReadOp
    : PolymorphicDialectOp<"read_const", [Pure, DeclareOpInterfaceMethods<
                                                    SymbolUserOpInterface>]> {
  let summary = "read a template parameter or expression";
  let description = [{
    This operation reads a named `poly.param` or `poly.expr` binding from the nearest enclosing
    `poly.template`. For a read nested in a `verif.contract` whose target is nested in a
    `poly.template`, it instead reads from the target's template. Its result type must match the
    result of a `poly.expr` or the restriction on a `poly.param`. For an unrestricted parameter,
    the result type states the type required at that use. A binding whose type is
    `!poly.tvar<...>` cannot be read directly.

    Examples:
    ```llzk
    // Assume the enclosing template declares @A : !felt.type and @B : index, and defines
    // @DoubleB as an index-typed poly.expr.
    // Read a `!felt.type` value from template parameter "@A".
    %0 = poly.read_const @A : !felt.type
    // Read an index value from template parameter "@B".
    %1 = poly.read_const @B : index
    // Read the index value yielded by template expression "@DoubleB".
    %2 = poly.read_const @DoubleB : index
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$const_name);
  let results = (outs ConstReadType:$val);

  let assemblyFormat = [{ $const_name `:` type($val) attr-dict }];
}

def LLZK_UnifiableCastOp : PolymorphicDialectOp<"unifiable_cast", [Pure]> {
  let summary = "cast between two unifiable types";
  let description = [{
    This operation represents a value with a different type, with the restriction that the input
    and output types of the cast are unifiable.

    Most operations that accept LLZK types also accept operands whose types unify, so an explicit
    cast is normally unnecessary. Use this operation when neither the input nor result type can be
    changed but MLIR still requires exact type equality. For example, a conversion pattern can use
    this operation instead of an `unrealized_conversion_cast` while preserving the requirement that
    the unequal types unify.

    When constant folding has a constant input and the input and result types are identical, this
    operation folds to that input. Merely unifiable unequal types do not fold because the
    replacement would have the wrong result type.

    Example:
    ```llzk
    %0 = some_other_op : !array.type<@N x !felt.type>
    %1 = poly.unifiable_cast %0 : (!array.type<@N x !felt.type>) -> !array.type<affine_map<()[s0, s1] -> (s0 + s1)> x !felt.type>
    ```
  }];

  let arguments = (ins AnyLLZKType:$input);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{
    $input `:` functional-type($input, results) attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def LLZK_ApplyMapOp : PolymorphicDialectOp<"applymap", [Pure]> {
  let summary = "apply an AffineMap";
  let description = [{
    This operation applies an AffineMap to a list of SSA values, yielding a single
    SSA value. The number of dimension and symbol arguments must be equal to the
    respective number of dimensional and symbolic inputs to the AffineMap; the
    AffineMap has to be one-dimensional, and so this operation always returns one
    value. The input operands and result all have `index` type.

    Named map example:
    ```llzk
    #map10 = affine_map<(d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)>
    ...
    %1 = poly.applymap(%s, %t) #map10
    ```

    Inline example:
    ```llzk
    %2 = poly.applymap(%42)[%n] affine_map<(i)[s0] -> (i+s0)>
    ```
  }];

  let arguments = (ins AffineMapAttr:$map, Variadic<Index>:$mapOperands,
      IndexAttr:$numDims);
  let results = (outs Index);

  // Define builders manually so inference of `numDims` attribute is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::mlir::AffineMapAttr":$map,
                                CArg<"::mlir::ValueRange", "{}">:$mapOperands),
                            [{
                  $_state.addOperands(mapOperands);
                  Properties &props = $_state.getOrAddProperties<Properties>();
                  props.setMap(map);
                  props.setNumDims($_builder.getIntegerAttr($_builder.getIndexType(),
                                                            map.getAffineMap().getNumDims()));
                  $_state.addTypes($_builder.getIndexType());
                }]>,
                  OpBuilder<(ins "::mlir::AffineMap":$map,
                                CArg<"::mlir::ValueRange", "{}">:$mapOperands),
                            [{
                  build($_builder, $_state, ::mlir::AffineMapAttr::get(map), mapOperands);
                }]>,
                  OpBuilder<(ins "::mlir::AffineExpr":$expr,
                                CArg<"::mlir::ValueRange", "{}">:$mapOperands),
                            [{
                  auto map = ::mlir::AffineMap::inferFromExprList({expr}, $_builder.getContext()).front();
                  build($_builder, $_state, map, mapOperands);
                }]>];

  let assemblyFormat = [{
    custom<DimAndSymbolList>($mapOperands, $numDims) $map attr-dict
  }];

  let hasVerifier = 1;
  let hasFolder = 1;

  let extraClassDeclaration = [{
    /// Returns the affine map to be applied by this operation.
    ::mlir::AffineMap inline getAffineMap() { return getMap(); }

    /// Returns the affine value map computed from this operation.
    ::mlir::affine::AffineValueMap getAffineValueMap() {
      return ::mlir::affine::AffineValueMap(getAffineMap(), getOperands(), getResult());
    }

    /// Returns all dimension operands.
    ::mlir::ValueRange getDimOperands() {
      return ::mlir::OperandRange{
                          getOperands().begin(),
                          getOperands().begin() + getMap().getNumDims()};
    }

    /// Returns all symbol operands.
    ::mlir::ValueRange getSymbolOperands() {
      return ::mlir::OperandRange{
                          getOperands().begin() + getMap().getNumDims(),
                          getOperands().end()};
    }
  }];
}

#endif // LLZK_POLYMORPHIC_OPS
