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

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class PolymorphicDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<PolymorphicDialect, mnemonic, traits>;

def LLZK_ConstReadOp
    : PolymorphicDialectOp<
          "read_const", [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "read value of a struct parameter";
  let description = [{
    This operation reads the value from the named constant parameter of
    the struct/component in which this op appears. The op itself puts
    some restriction on the type of this value, but leaves it to a later
    type-checking pass to ensure the struct parameters are instantiated
    with types matching the uses of the parameter within the struct.

    Examples:
    ```llzk
    // Read a `!felt.type` value from struct parameter "@A"
    %0 = poly.read_const @A : !felt.type
    // Read a value from struct parameter "@B" where its type is
    // specified by struct parameter "@T"
    %1 = poly.read_const @B : !poly.tvar<@T>
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$const_name);
  let results = (outs ConstReadType:$val);

  let assemblyFormat = [{ $const_name `:` type($val) attr-dict }];
}

def LLZK_UnifiableCastOp : PolymorphicDialectOp<"unifiable_cast", [Pure]> {
  let summary = "cast between two unifiable types";
  let description = [{
    This operation reinterprets a value as a different type with the restriction
    that the input and output types of the cast are unifiable.

    Most ops that accept LLZK types accept unifiable types as input and thus there
    is no need for casting between types. This op is meant to be used in situations where
    is not possible to modify the given or the target type and they are different but unifiable.
    For example, inside a conversion pattern the driver may introduce `unrealized_conversion_cast`
    operations if the types are not equal. This will happen regardless of whether the two types unify.
    This cast can be introduced instead of the default cast operation to satisfy MLIR's assumptions
    on type equality.

    Example:
    ```llzk
    %0 = some_other_op : !array.type<@N x !felt.type>
    %1 = unifiable_cast %0 : (!array.type<@N x @felt.type>) -> !array.type<affine_map<()[s0, s1] -> (s0 + s1)> x !felt.type>
    ```
  }];

  let arguments = (ins AnyLLZKType:$input);
  let results = (outs AnyLLZKType:$result);
  let assemblyFormat = [{
    $input `:` functional-type($input, results) attr-dict
  }];

  let hasVerifier = 1;
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
