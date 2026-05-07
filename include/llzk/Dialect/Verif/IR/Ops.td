//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_ARRAY_OPS
#define LLZK_ARRAY_OPS

include "llzk/Dialect/Array/IR/Dialect.td"
include "llzk/Dialect/Array/IR/OpInterfaces.td"
include "llzk/Dialect/Array/IR/Types.td"
include "llzk/Dialect/Shared/OpTraits.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class ArrayDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<ArrayDialect, mnemonic, traits>;

class ArrayAccessOpBase<string mnemonic, list<Trait> traits = []>
    : ArrayDialectOp<
          mnemonic,
          traits#[DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                  DeclareOpInterfaceMethods<ArrayAccessOpInterface>]> {
  let extraClassDeclaration = [{
    /// Gets the type of the referenced base array.
    inline ::llzk::array::ArrayType getArrRefType() {
      return ::llvm::cast<ArrayAccessOpInterface>(getOperation()).getArrRefType();
    }
  }];
}

// isRead: read(1) vs write(0) ops
class ScalarArrayAccessOp<string mnemonic, bit isRead, list<Trait> traits = []>
    : ArrayAccessOpBase<
          mnemonic,
          traits#[DeclareOpInterfaceMethods<DestructurableAccessorOpInterface>,
                  DeclareOpInterfaceMethods<PromotableMemOpInterface>]> {
  let extraClassDefinition = [{
    /// Required by DestructurableAllocationOpInterface / SROA pass
    bool $cppClass::canRewire(const ::mlir::DestructurableMemorySlot &slot,
                ::llvm::SmallPtrSetImpl<::mlir::Attribute> &usedIndices,
                ::mlir::SmallVectorImpl<::mlir::MemorySlot> &mustBeSafelyUsed,
                const ::mlir::DataLayout &dataLayout) {
      return ::llvm::cast<ArrayAccessOpInterface>(getOperation())
                .canRewire(slot, usedIndices, mustBeSafelyUsed, dataLayout);
    }

    /// Required by DestructurableAllocationOpInterface / SROA pass
    ::mlir::DeletionKind $cppClass::rewire(const ::mlir::DestructurableMemorySlot &slot,
                ::llvm::DenseMap<::mlir::Attribute, ::mlir::MemorySlot> &subslots,
                ::mlir::OpBuilder &builder, const ::mlir::DataLayout &dataLayout) {
      return ::llvm::cast<ArrayAccessOpInterface>(getOperation())
                .rewire(slot, subslots, builder, dataLayout);
    }

    /// Required by PromotableMemOpInterface / mem2reg pass
    bool $cppClass::loadsFrom(const ::mlir::MemorySlot &slot) {
      return }]#!if(isRead, "getArrRef() == slot.ptr", "false")#[{;
    }

    /// Required by PromotableMemOpInterface / mem2reg pass
    bool $cppClass::storesTo(const ::mlir::MemorySlot &slot) {
      return }]#!if(isRead, "false", "getArrRef() == slot.ptr")#[{;
    }

    /// Required by PromotableAllocationOpInterface / mem2reg pass
    ::mlir::Value $cppClass::getStored(const ::mlir::MemorySlot &, ::mlir::OpBuilder &,
                                       ::mlir::Value, const ::mlir::DataLayout &) {
      }]#!if(isRead,
             "llvm_unreachable(\"getStored() should not be called on "
             "$cppClass\")",
             "return getRvalue()")#[{;
    }

    /// Return `true` if the op is a read, `false` if it's a write.
    bool $cppClass::isRead() {
      return }]#!if(isRead, "true", "false")#[{;
    }
  }];
}

// isRead: read(1) vs write(0) ops
class RangedArrayAccessOp<string mnemonic, bit isRead, list<Trait> traits = []>
    : ArrayAccessOpBase<mnemonic, traits> {
  let extraClassDefinition = [{
    /// Return `true` if the op is a read, `false` if it's a write.
    bool $cppClass::isRead() {
      return }]#!if(isRead, "true", "false")#[{;
    }
  }];
}

//===------------------------------------------------------------------===//
// Array operations
//===------------------------------------------------------------------===//

def LLZK_CreateArrayOp
    : ArrayDialectOp<
          "new",
          [Pure, AttrSizedOperandSegments, VerifySizesForMultiAffineOps<1>,
           DeclareOpInterfaceMethods<SymbolUserOpInterface>,
           DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
           DeclareOpInterfaceMethods<PromotableAllocationOpInterface>,
           DeclareOpInterfaceMethods<DestructurableAllocationOpInterface>,
           VariadicTypesMatchWith<
               "operand types match result type", "result", "elements",
               "resultTypeToElementsTypes($_self)", "std::equal_to<>()">]> {
  let summary = "create a new array";
  let description = [{
    This operation creates a new array with the given elements.
    The arguments are passed as a flat array but get arranged in
    row-major order according the shape declared in the type.

    Examples:
    ```llzk
    %0 = array.new %a, %b, %c : !array.type<3 x !felt.type>

    // Create an array of the specified shape, initialized with the given
    // values assigned in row-major order: [[%a, %b], [%c, %d]]
    %1 = array.new %a, %b, %c, %d : !array.type<2,2 x !felt.type>

    // Create an uninitialized array: [[?, ?], [?, ?], [?, ?]]
    %2 = array.new : !array.type<3,2 x !felt.type>
    ```

    The values used to construct the array must have type that exactly matches
    the element type of the specified array type. This is true even if a `tvar`
    type is used. In other words, cannot mix `tvar<@X>` with `tvar<@Y>` or any
    concrete type. In such a scenario, first create an uninitialized array, as
    shown in the examples above, and then use `array.write` to write each element
    of the array.

    Implementation note: This restriction exists due to a combination of:
    (1) we have chosen to infer the type of `$elements` from the `$result`
    ArrayType, via parseInferredArrayType(), rather than requiring the type of
    every element be listed in the assembly format and,
    (2) within the parser for an Op, there is no way to get the Value instances
    for the operands aside from `OpAsmParser::resolveOperands()` which requires
    the type of every operand to be known and ends up comparing the expected
    to actual type via `operator==`. Thus, there is no way for this to be
    successful apart from all elements having the exact type inferred in (1).

    Also note that `std::equal_to` is used in the `VariadicTypesMatchWith`
    trait on this Op so that the verifier function mirrors the aforementioned
    restriction in the parser.

    In some cases, the length of an uninitialized array depends on the value
    of the loop induction variable (i.e., each iteration creates an array with
    a different size/shape). In that case, an AffineMapAttr can be used to
    specify the dimension size in the ArrayType and the optional instantiation
    parameter list of this operation must be used to instatiate all AffineMap
    used in the array dimensions.

    Examples:
    ```llzk
    // Create an uninitialized array with dimension size defined by AffineMap
    #IdxToLen = affine_map<(i) -> (5*i+1)>
    %3 = array.new {(%i)} : !array.type<#IdxToLen x index>

    // Create an uninitialized array with multiple dimensions defined by
    //  AffineMap. The list of instantiation parameters are assigned to
    //  the AffineMap dimensions left-to-right.
    #M1 = affine_map<(i)[c] -> (c+i)>
    #M3 = affine_map<(m,n) -> (5*m+n)>
    %4 = array.new{(%i)[%c], (%m,%n)} : !array.type<#M1,2,#M3 x i1>
    ```
  }];

  // See `VerifySizesForMultiAffineOps` for more explanation of these arguments.
  let arguments = (ins
      // Elements to initialize the array. Length is either zero or equal to
      // the length of the flattened/linearized result ArrayType.
      Variadic<ArrayElemType>:$elements,
      // List of AffineMap operand groups where each group provides the
      // arguments to instantiate the next (left-to-right) AffineMap used as an
      // array dimension in the result ArrayType.
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      // Within each group in '$mapOperands', denotes the number of values that
      // are AffineMap "dimensional" arguments with the remaining values being
      // AffineMap "symbolic" arguments.
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      // Denotes the size of each variadic group in '$mapOperands'.
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs LLZK_ArrayType:$result);

  // Define builders manually so inference of operand layout attributes is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::llzk::array::ArrayType":$result,
                      CArg<"::mlir::ValueRange", "{}">:$elements)>,
                  OpBuilder<(ins "::llzk::array::ArrayType":$result,
                      "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                      "::mlir::DenseI32ArrayAttr":$numDimsPerMap)>,
                  OpBuilder<
                      (ins "::llzk::array::ArrayType":$result,
                          "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                          "::llvm::ArrayRef<int32_t>":$numDimsPerMap),
                      [{
                        build($_builder, $_state, result, mapOperands, odsBuilder.getDenseI32ArrayAttr(numDimsPerMap));
                      }]>];

  // This uses the custom parseInferredArrayType function to compute the type
  //  of '$elements' to match the type of '$result', except when '$elements'
  //  is empty, then the type of '$elements' must also be empty (size == 0).
  // The if-then-else has '$elements' second so that an empty '$elements' list
  //  can be parsed when neither of these is specified.
  let assemblyFormat = [{
        ( `{` custom<MultiDimAndSymbolList>($mapOperands, $numDimsPerMap)^ `}` ) : ( $elements )?
        `:` type($result)
        `` custom<InferredArrayType>(type($elements), ref($elements), ref(type($result)))
        custom<AttrDictWithWarnings>(attr-dict, prop-dict)
      }];

  let extraClassDeclaration = [{
  private:
    static ::mlir::ParseResult parseInferredArrayType(::mlir::OpAsmParser &parser,
        ::llvm::SmallVector<::mlir::Type,1> &elementsTypes,
        ::mlir::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> elements,
        ::mlir::Type resultType
    );

    static void printInferredArrayType(::mlir::OpAsmPrinter &printer, CreateArrayOp,
        ::mlir::TypeRange, ::mlir::OperandRange, ::mlir::Type
    );

    static ::llvm::SmallVector<::mlir::Type> resultTypeToElementsTypes(::mlir::Type resultType);
  }];

  let hasVerifier = 1;
}

def LLZK_ReadArrayOp
    : ScalarArrayAccessOp<
          "read", true, [ArrayTypeElemsUnifyWithResultCustomInfer<"arr_ref">]> {
  let summary = "read scalar from an array";
  let description = [{
    This operation reads the value from an array at the specified position.

    Example of 1-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %0 = array.new %a, %b, %c : !array.type<3 x !felt.type>
    // %1 is now equal to %a
    %1 = array.read %0[%i] : !array.type<3 x !felt.type>, !felt.type
    ```

    Example of 3-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %j = arith.constant 1 : index
    %k = arith.constant 4 : index
    %0 = array.new ... : !array.type<3,10,15 x !felt.type>
    // %1 is now equal to %a
    %1 = array.read %0[%i, %j, %k] : !array.type<3,10,15 x !felt.type>, !felt.type
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices);
  let results = (outs ArrayElemType:$result);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `:` type($arr_ref) `,` type($result) attr-dict
  }];

  let hasVerifier = 1;
}

def LLZK_WriteArrayOp
    : ScalarArrayAccessOp<
          "write", false, [ArrayElemTypeUnifyWith<"arr_ref", "rvalue">]> {
  let summary = "write scalar to an array";
  let description = [{
    This operation writes a value into an array at the specified position.

    Example of 1-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %0 = felt.const 42
    %1 = array.new %a, %b, %c : !array.type<3 x !felt.type>
    // The array now is [%0, %b, %c]
    array.write %1[%i] = %0 : !array.type<3 x !felt.type>, !felt.type
    ```

    Example of 2-dimensional array:
    ```llzk
    %i = arith.constant 0 : index
    %j = arith.constant 0 : index
    %0 = felt.const 42
    %1 = array.new %a, %b, %c, %d : !array.type<2,2 x !felt.type>
    // The array now is [[%0, %b], [%c, %d]]
    array.write %1[%i, %j] = %0 : !array.type<2,2 x !felt.type>, !felt.type
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices,
      ArrayElemType:$rvalue);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `=` $rvalue `:` type($arr_ref) `,` type($rvalue) attr-dict
  }];

  let hasVerifier = 1;
}

def LLZK_ExtractArrayOp
    : RangedArrayAccessOp<"extract",
                          true, [InferTypeOpAdaptorWithIsCompatible]> {
  let summary = "read subarray from a multi-dimensional array";
  let description = [{
    This operation takes an N-dimensional array and K indices and extracts the
    (N-K)-dimensional array by applying the given indices to the first `K`
    dimensions of the array. Error if `K >= N`. Use `array.read` instead if `K == N`.

    Extracting a 1-D array from 3-D array by selecting the index of 2 dimensions:
    ```llzk
    %i = arith.constant 1 : index
    %0 = array.new ... : !array.type<3,10,15 x !felt.type>
    %1 = array.extract %0[%i,%i] : !array.type<3,10,15 x !felt.type>, !array.type<15 x !felt.type>
    ```

    Extracting 1-D arrays for subcomponents:
    ```llzk
    scf.for %iv = %lb to %up step %step {
      %p = array.extract %in[%iv] : !array.type<@N,2 x !felt.type>
      %c = array.read %a[%iv] : !array.type<@N x !struct.type<@SubC>>, !struct.type<@SubC>
      function.call @SubC::@constrain(%c, %p) : (!struct.type<@SubC>, !array.type<2 x !felt.type>) -> ()
    }
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices);
  let results = (outs LLZK_ArrayType:$result);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `:` type($arr_ref) attr-dict
  }];
}

def LLZK_InsertArrayOp : RangedArrayAccessOp<"insert", false> {
  let summary = "write subarray into a multi-dimensional array";
  let description = [{
    This operation takes an N-dimensional array, K indices, and an (N+K)-dimensional
    array and inserts the N-dimensional array into the (N+K)-dimensional array at the
    position specified by applying the given indices to the first `K` dimensions of
    the (N+K)-dimensional array. Use `array.write` instead if `N == 0` (LLZK array type
    must have at least 1 dimension so a 0-dimensional array cannot exist anyway).

    Inserting 1-D arrays into a 2-D array:
    ```llzk
    %c = array.new : !array.type<2,3 x index>
    // Array %c is uninitialized [[?, ?, ?], [?, ?, ?]]
    %0 = arith.constant 0 : index
    %a = array.new %r, %s, %t : !array.type<3 x index>
    array.insert %c[%0] = %a : !array.type<2,3 x index>, !array.type<3 x index>
    // Array %c is now [[%r, %s, %t], [?, ?, ?]]
    %1 = arith.constant 1 : index
    %b = array.new %x, %y, %z : !array.type<3 x index>
    array.insert %c[%1] = %b : !array.type<2,3 x index>, !array.type<3 x index>
    // Array %c is now [[%r, %s, %t], [%x, %y, %z]]
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Variadic<Index>:$indices,
      LLZK_ArrayType:$rvalue);

  let assemblyFormat = [{
    $arr_ref `[` $indices `]` `=` $rvalue `:` type($arr_ref) `,` type($rvalue) attr-dict
  }];

  let hasVerifier = 1;
}

def LLZK_ArrayLengthOp
    : ArrayDialectOp<"len", [Pure,
                             DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                             DeclareOpInterfaceMethods<ArrayRefOpInterface>]> {
  let summary = "return the length of an array";
  let description = [{
    This operation returns the size of the specified dimension of an array.

    Example:
    ```llzk
    %a = array.new : !array.type<2,3 x !felt.type>
    %0 = arith.constant 0 : index
    %x = array.len %a, %0 : !array.type<2,3 x !felt.type> // result is 2
    %1 = arith.constant 1 : index
    %y = array.len %a, %1 : !array.type<2,3 x !felt.type> // result is 3
    ```
  }];

  let arguments = (ins LLZK_ArrayType:$arr_ref, Index:$dim);
  let results = (outs Index:$length);

  let assemblyFormat = [{ $arr_ref `,` $dim `:` type($arr_ref) attr-dict }];

  let extraClassDeclaration = [{
    /// Gets the type of the referenced base array.
    inline ::llzk::array::ArrayType getArrRefType() {
      return ::llvm::cast<ArrayRefOpInterface>(getOperation()).getArrRefType();
    }
  }];
}

#endif // LLZK_ARRAY_OPS
