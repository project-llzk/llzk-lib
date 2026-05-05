//===-- Types.td -------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ARRAY_TYPES
#define LLZK_ARRAY_TYPES

include "llzk/Dialect/Array/IR/Dialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def ArrayElemType : Type<CPred<"::llzk::isValidArrayElemType($_self)">,
                         "a valid array element type">;

def LLZK_ArrayType
    : TypeDef<ArrayDialect, "Array",
              [ShapedTypeInterface,
               DeclareTypeInterfaceMethods<DestructurableTypeInterface>]> {
  let mnemonic = "type";
  let summary = "n-dimensional array";
  let description = [{
    Array type with a ranked shape and homogeneous element type.
    It can only be instantiated with the following element types:
      - Any LLZK type other than an ArrayType (i.e., arrays cannot be nested)
      - IndexType
      - Unsigned integers of 1 bit (aka booleans)

    The dimensions of the array are specified using a list of attributes, one
    per dimension. Each attribute must be one of the following:
      - IntegerAttr (with IndexType), specifying a fixed dimension size
      - SymbolRefAttr, specifying a dimension size defined by a struct parameter or
        global constant
      - AffineMapAttr, specifying a dimension size computed from surrounding loop
        induction variables

    ```llzk
    // Example array of 5 by 2 elements of `Felt` type.
    !array.type<5,2 x !felt.type>

    // Example array using a struct parameter for one dimension.
    !array.type<5,@A x index>
    ```
  }];

  let parameters =
      (ins TypeParameter<"::mlir::Type",
                         "Type of all elements within the array.">:$elementType,
          ArrayRefParameter<
              "::mlir::Attribute",
              "List of array dimension size specifiers.">:$dimensionSizes,
          ArrayRefParameter<"int64_t",
                            "Array shape, for ShapedTypeInterface, computed "
                            "from `$dimensionSizes`.">:$shape);

  // The custom<DerivedShape>() section is 0-length (it is used to compute the
  // "$shape" attribute from the "$dimensionSizes" attribute) and the `` before
  // it avoids excess space in the format. Additionally, it is placed right
  // after parsing "$dimensionSizes" so the source location pointer in the
  // parser remains at "$dimensionSizes" in case an error occurs during the
  // conversion.
  let assemblyFormat = [{
    `<` custom<AttrVec>($dimensionSizes)
    `` custom<DerivedShape>($shape, ref($dimensionSizes))
    `x` $elementType `>`
  }];

  let genVerifyDecl = 1;

  let skipDefaultBuilders = 1;
  let builders =
      [TypeBuilderWithInferredContext<
           (ins "::mlir::Type":$elementType,
               "::llvm::ArrayRef<::mlir::Attribute>":$dimensionSizes),
           [{
      assert(elementType && "element type cannot be null");
      ::mlir::MLIRContext *ctx = elementType.getContext();
      OwningEmitErrorFn emitErrorFn = ::llzk::wrapNullableInFlightDiagnostic(emitError, ctx);
      auto dimSizesRes = forceIntAttrTypes(dimensionSizes, emitErrorFn);
      if(::mlir::failed(dimSizesRes)) { return ArrayType(); }
      ::llvm::SmallVector<::mlir::Attribute> dimSizes = *dimSizesRes;
      ::llvm::SmallVector<int64_t> shape;
      ::mlir::LogicalResult res = computeShapeFromDims(emitErrorFn, dimSizes, shape);
      if(::mlir::failed(res)) { return ArrayType(); }
      return $_get(ctx, elementType, dimSizes, shape);
    }]>,
       TypeBuilderWithInferredContext<(ins "::mlir::Type":$elementType,
                                          "::llvm::ArrayRef<int64_t>":$shape),
                                      [{
      assert(elementType && "element type cannot be null");
      ::mlir::MLIRContext *ctx = elementType.getContext();
      ::llvm::SmallVector<::mlir::Attribute> dimSizes;
      ::mlir::LogicalResult res = computeDimsFromShape(ctx, shape, dimSizes);
      if(::mlir::failed(res)) { return ArrayType(); }
      return $_get(ctx, elementType, dimSizes, shape);
    }]>];

  let extraClassDeclaration = [{
  private:
    /// Call the given function for each valid index for this ArrayType.
    /// Return `false` if this ArrayType does not have a static shape.
    bool collectIndices(::llvm::function_ref<void(::mlir::ArrayAttr)>) const;

  public:
    /// Return a list of all valid indices for this ArrayType.
    std::optional<::llvm::SmallVector<::mlir::ArrayAttr>> getSubelementIndices() const;

    /// Returns if this type is ranked, i.e., it has a known number of dimensions.
    /// LLZK arrays are always ranked, i.e., the number of dimensions is known.
    /// Required by the ShapedTypeInterface interface.
    inline bool hasRank() const { return true; }

    /// Clone this type with the given shape and element type. If the
    /// provided shape is `std::nullopt`, the current shape of the type is used.
    /// Required by the ShapedTypeInterface interface.
    ArrayType cloneWith(std::optional<::llvm::ArrayRef<int64_t>> shape, ::mlir::Type elementType) const;

    /// Clone this type with the given dimensions and element type. If the provided
    /// dimensions are `std::nullopt`, the current dimensions of the type are used.
    /// Note: This is preferred over cloneWith(..int64_t..) because this Attribute
    /// version can carry more information than the `int64_t` which must default to
    /// `kDynamic` when the Attribute is anything other than an integer constant.
    ArrayType cloneWith(::mlir::Type elementType, std::optional<::llvm::ArrayRef<::mlir::Attribute>> dimensions = std::nullopt) const;
  }];

  let extraClassDefinition = [{
    namespace {
      /// This definition of `emitError` is used by the `get()` functions generated by the
      /// custom builders for this type. The `getChecked()` functions generated by those same
      /// builders have a parameter with this same name that shadows this definition so the
      /// `getChecked()` versions will use the function supplied via the parameter. Regardless,
      /// `wrapNullableInFlightDiagnostic()` checks for `nullptr` and generates a default if
      /// necessary. This approach, although a bit hacky, allows a legitimate error function to
      /// be used whenever available, only reverting to a default in the `get()` function.
      const ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError = nullptr;
    }
  }];
}

#endif // LLZK_ARRAY_TYPES
