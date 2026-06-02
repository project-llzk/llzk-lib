//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from mlir/include/mlir/Dialect/Func/IR/FuncOps.td
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_FUNC_OPS
#define LLZK_FUNC_OPS

include "llzk/Dialect/Function/IR/Dialect.td"
include "llzk/Dialect/Shared/OpTraits.td"
include "llzk/Dialect/Shared/Types.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class FunctionDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<FunctionDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// FuncDefOp
//===----------------------------------------------------------------------===//

def FuncDefOp
    : FunctionDialectOp<
          "def",
          [ParentOneOf<["::mlir::ModuleOp", "::llzk::component::StructDefOp",
                        "::llzk::polymorphic::TemplateOp"]>,
           DeclareOpInterfaceMethods<SymbolUserOpInterface>, AffineScope,
           AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove]> {
  // NOTE: Cannot have SymbolTable trait because that would cause global
  // functions without a body to produce "Operations with a 'SymbolTable' must
  // have exactly one block"
  let summary = "An operation with a name containing a single `SSACFG` region";
  let description = [{
    Operations within the function cannot implicitly capture values defined
    outside of the function, i.e., functions are `IsolatedFromAbove`. All
    external references must use function arguments (which are passed by value)
    or reference external members or globals by symbol name.

    Functions appearing within a `struct.def` have specific semantics and must
    be named `compute`, `constrain`, or `product`. Functions appearing at the
    module level (i.e. not within a `struct.def`) have no name restrictions and
    their body may be elided to denote an external function declaration.

    Modules and `struct.def` ops are not allowed to be nested within functions.

    Function arguments may carry an optional `function.arg_name` attribute to
    preserve the source-level argument name independently from the printed SSA
    block argument name. The value must be a non-empty, untyped `StringAttr`,
    and all attached `function.arg_name` values must be unique within the
    function. Typed string attributes such as `"x" : i1` are rejected. The
    attribute is only valid on function arguments.
    Argument-splitting transforms preserve this metadata by deriving unique
    names for generated arguments, such as `input[0]` for array elements or
    `self.member` for struct members.

    Function results may similarly carry an optional `function.res_name`
    attribute. The value must be a non-empty, untyped `StringAttr`, all attached
    result names must be unique within the function, and the attribute is only
    valid on function results.

    Example:

    ```llzk
    // External function definitions.
    function.def private @abort()
    function.def private @scribble(
      !array.type<5 x !felt.type> {function.arg_name = "input"},
      !struct.type<@Hello> {function.arg_name = "state"}) -> i1

    // A function that returns its argument twice:
    function.def @count(%x: !felt.type {function.arg_name = "x"})
        -> (!felt.type {function.res_name = "first"},
            !felt.type {function.res_name = "second"}) {
      function.return %x, %x: !felt.type, !felt.type
    }

    // Function definition within a component
    struct.def @NonZero {
      function.def @compute(%a: !felt.type {function.arg_name = "a"}) { function.return }
      function.def @constrain(%a: !felt.type {function.arg_name = "a"}) { function.return }
    }
    ```
  }];

  // Duplicated from the pre-defined `func` dialect. We don't store the
  // visibility attribute but, since we use `function_interface_impl` for
  // parsing/printing, there is still the requirement that global functions
  // declared without a body must specify the `private` visibility.
  // Additionally, the default parsing/printing functions allow attributes on
  // the arguments, results, and function itself.
  //    ```llzk
  //    // Argument attribute
  //    function.def private @example_fn_arg(%x: i1 {llzk.pub})
  //    function.def private @example_fn_arg_name(%x: i1 {function.arg_name =
  //    "x"})
  //
  //    // Result attribute
  //    function.def @example_fn_result() -> (i1 {dialectName.attrName = 0 :
  //    i1})
  //
  //    // Function attribute
  //    function.def @example_fn_attr() attributes {dialectName.attrName =
  //    false}
  //    ```
  let arguments = (ins SymbolNameAttr:$sym_name,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins "::llvm::StringRef":$name,
      "::mlir::FunctionType":$type,
      CArg<"::llvm::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs,
      CArg<"::llvm::ArrayRef<::mlir::DictionaryAttr>", "{}">:$argAttrs)>];

  let extraClassDeclaration = [{
    static FuncDefOp create(::mlir::Location location, ::llvm::StringRef name, ::mlir::FunctionType type,
                         ::llvm::ArrayRef<::mlir::NamedAttribute> attrs = {});
    static FuncDefOp create(::mlir::Location location, ::llvm::StringRef name, ::mlir::FunctionType type,
                         ::mlir::Operation::dialect_attr_range attrs);
    static FuncDefOp create(::mlir::Location location, ::llvm::StringRef name, ::mlir::FunctionType type,
                         ::llvm::ArrayRef<::mlir::NamedAttribute> attrs,
                         ::llvm::ArrayRef<::mlir::DictionaryAttr> argAttrs);

    /// Create a deep copy of this function and all of its blocks, remapping any
    /// operands that use values outside of the function using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for function arguments, these arguments are not
    /// included in the new function. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    FuncDefOp clone(::mlir::IRMapping &mapper);
    FuncDefOp clone();

    /// Clone the internal blocks and attributes from this function into dest.
    /// Any cloned blocks are appended to the back of dest. This function
    /// asserts that the attributes of the current function and dest are
    /// compatible.
    void cloneInto(FuncDefOp dest, ::mlir::IRMapping &mapper);

    /// Return `true` iff the function def has the `allow_constraint` attribute.
    inline bool hasAllowConstraintAttr() {
      return getOperation()->hasAttr(llzk::function::AllowConstraintAttr::name);
    }

    /// Add (resp. remove) the `allow_constraint` attribute to (resp. from) the function def.
    void setAllowConstraintAttr(bool newValue = true);

    /// Return `true` iff the function def has the `allow_witness` attribute.
    inline bool hasAllowWitnessAttr() {
      return getOperation()->hasAttr(llzk::function::AllowWitnessAttr::name);
    }

    /// Add (resp. remove) the `allow_witness` attribute to (resp. from) the function def.
    void setAllowWitnessAttr(bool newValue = true);

    /// Return `true` iff the function def has the `allow_non_native_field_ops` attribute.
    inline bool hasAllowNonNativeFieldOpsAttr() {
      return getOperation()->hasAttr(llzk::function::AllowNonNativeFieldOpsAttr::name);
    }

    /// Add (resp. remove) the `allow_non_native_field_ops` attribute to (resp. from) the function def.
    void setAllowNonNativeFieldOpsAttr(bool newValue = true);

    /// Return `true` iff the argument at the given index has `pub` attribute.
    bool hasArgPublicAttr(unsigned index);

    /// Return `true` iff the argument at the given index has a `function.arg_name` attribute.
    bool hasArgName(unsigned index);

    /// Return the `function.arg_name` attribute for the argument at the given index.
    ::std::optional<::mlir::StringAttr> getArgNameAttr(unsigned index);

    /// Set the `function.arg_name` attribute for the argument at the given index.
    void setArgNameAttr(unsigned index, const ::mlir::StringAttr &attr);

    /// Set the `function.arg_name` attribute for the argument at the given index from a string.
    void setArgName(unsigned index, ::llvm::StringRef name);

    /// Required by FunctionOpInterface.
    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Required by FunctionOpInterface.
    /// Returns the argument types of this function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Required by FunctionOpInterface.
    /// Returns the result types of this function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Required by SymbolOpInterface.
    bool isDeclaration() { return isExternal(); }

    /// Return the full name for this function from the root module, including
    /// all surrounding symbol table names (i.e., modules and structs).
    ::mlir::SymbolRefAttr getFullyQualifiedName(bool requireParent = true);

    /// Return `true` iff the function name is `FUNC_NAME_COMPUTE` (if needed, a check
    /// that this FuncDefOp is located within a StructDefOp must be done separately).
    inline bool nameIsCompute() { return FUNC_NAME_COMPUTE == getSymName(); }

    /// Return `true` iff the function name is `FUNC_NAME_CONSTRAIN` (if needed, a
    /// check that this FuncDefOp is located within a StructDefOp must be done separately).
    inline bool nameIsConstrain() { return FUNC_NAME_CONSTRAIN == getSymName(); }

    /// Return `true` iff the function name is `FUNC_NAME_PRODUCT` (if needed, a
    /// check that this FuncDefOp is located within a StructDefOp must be done separately).
    inline bool nameIsProduct() { return FUNC_NAME_PRODUCT == getSymName(); }

    /// Return `true` iff the function is within a StructDefOp
    inline bool isInStruct() { return ::llzk::component::isInStruct(*this); }

    /// Return `true` iff the function is within a StructDefOp and named `FUNC_NAME_COMPUTE`.
    inline bool isStructCompute() { return isInStruct() && nameIsCompute(); }

    /// Return `true` iff the function is within a StructDefOp and named `FUNC_NAME_CONSTRAIN`.
    inline bool isStructConstrain() { return isInStruct() && nameIsConstrain(); }

    /// Return `true` iff the function is within a StructDefOp and named `FUNC_NAME_PRODUCT`.
    inline bool isStructProduct() { return isInStruct() && nameIsProduct(); }

    /// Return the "self" value (i.e. the return value) from the function (which must be
    /// named `FUNC_NAME_COMPUTE`).
    ::mlir::Value getSelfValueFromCompute();

    /// Return the "self" value (i.e. the first parameter) from the function (which must be
    /// named `FUNC_NAME_CONSTRAIN`).
    ::mlir::Value getSelfValueFromConstrain();

    /// Assuming the name is `FUNC_NAME_COMPUTE`, return the single StructType result.
    ::llzk::component::StructType getSingleResultTypeOfCompute();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp
    : FunctionDialectOp<"return", [HasParent<"::llzk::function::FuncDefOp">,
                                   Pure, MemRefsNormalizable, ReturnLike,
                                   Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `function.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```llzk
    function.def @foo() : (!felt.type, index) {
      ...
      return %0, %1 : !felt.type, index
    }
    ```
  }];

  let arguments = (ins Variadic<AnyLLZKType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

def CallOp : FunctionDialectOp<
                 "call", [MemRefsNormalizable, AttrSizedOperandSegments,
                          VerifySizesForMultiAffineOps<1>,
                          DeclareOpInterfaceMethods<CallOpInterface>,
                          DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `function.call` operation represents a call to another function. The operands
    and result types of the call must match the specified function type. The
    callee is encoded as a symbol reference attribute named "callee" which must
    be the full path to the target function from the root module (i.e., the module
    containing the [llzk::LANG_ATTR_NAME] attribute).

    Example:
    ```llzk
    // Call a global function defined in the root module.
    function.call @do_stuff(%0) : (!struct.type<@Bob>) -> ()
    %1, %2 = function.call @split(%x) : (index) -> (index, index)

    // Call a function within a component
    %2 = function.call @OtherStruct::@compute(%3, %4) : (index, index) -> !struct.type<@OtherStruct>
    function.call @OtherStruct::@constrain(%5, %6) : (!struct.type<@OtherStruct>, !felt.type) -> ()
    ```

    When the return StructType of a `compute()` function uses AffineMapAttr to
    express struct parameter(s) that depend on a loop variable, the optional
    instantiation parameter list of this operation must be used to instatiate
    all AffineMap used as parameters to the StructType.

    Examples:
    ```llzk
    #M = affine_map<(i)[] -> (5*i+1)>
    %r = function.call @A::@compute(%x){(%i)} : (!felt.type) -> !struct.type<@A<[#M]>>
    ```

    When the call targets a free function within a `poly.template` region, the optional
    template parameter list can be used to instantiate all `poly.param` symbols within
    the template. If all `poly.param` symbols are used within the function signature,
    this can be elided. Otherwise, it is required to instantiate the function. The `?`
    wildcard can be used for any `poly.param` with a `poly.tvar` type restriction, even
    those that cannot be inferred from the function signature. The wildcard allows for
    inference of the type within the function body itself during the flattening pass
    but may fail if the type cannot be inferred from the function body.
  }];

  // See `VerifySizesForMultiAffineOps` for more explanation of these arguments.
  let arguments = (ins
      // Call target function reference.
      SymbolRefAttr:$callee,
      // List of arguments to call the target function.
      Variadic<AnyLLZKType>:$argOperands,
      // List of parameters to instantiate all `poly.param` symbols when the
      // callee is a free function inside a `poly.template` region.
      OptionalAttr<ArrayAttr>:$templateParams,
      // List of AffineMap operand groups where each group provides the
      // arguments to instantiate the next (left-to-right) AffineMap used as a
      // struct parameter in the result StructType.
      VariadicOfVariadic<Index, "mapOpGroupSizes">:$mapOperands,
      // Within each group in '$mapOperands', denotes the number of values that
      // are AffineMap "dimensional" arguments with the remaining values being
      // AffineMap "symbolic" arguments.
      DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$numDimsPerMap,
      // Denotes the size of each variadic group in '$mapOperands'.
      DenseI32ArrayAttr:$mapOpGroupSizes);
  let results = (outs Variadic<AnyLLZKType>);

  let assemblyFormat = [{
    $callee
    ( `<` custom<TemplateParams>($templateParams)^ `>` )?
    `` `(` $argOperands `)`
    ( `{` custom<MultiDimAndSymbolList>($mapOperands, $numDimsPerMap)^ `}` )?
    `:` functional-type($argOperands, results)
    custom<AttrDictWithWarnings>(attr-dict, prop-dict)
  }];

  let useCustomPropertiesEncoding = 1;

  // NOTE: In CreateArrayOp, the `verify()` function is declared in order to
  // call `verifyAffineMapInstantiations()`. However, in this op that check must
  // happen within `verifySymbolUses()` instead because the target FuncDefOp
  // must be resolved to determine if a target function named
  // "compute"/"constrain" is defined within a StructDefOp or within a ModuleOp
  // because the verification differs for those cases.

  // Define builders manually so inference of operand layout attributes is not
  // circumvented.
  let skipDefaultBuilders = 1;
  let builders =
      [OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::SymbolRefAttr":$callee,
           CArg<"::mlir::ValueRange", "{}">:$argOperands,
           CArg<"::llvm::ArrayRef<::mlir::Attribute>", "{}">:$templateParams)>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
           "::mlir::SymbolRefAttr":$callee,
           "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
           "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
           CArg<"::mlir::ValueRange", "{}">:$argOperands,
           CArg<"::llvm::ArrayRef<::mlir::Attribute>", "{}">:$templateParams)>,
       OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                     "::mlir::SymbolRefAttr":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state, resultTypes, callee, mapOperands,
                          $_builder.getDenseI32ArrayAttr(numDimsPerMap),
                          argOperands, templateParams);
                  }]>,
       OpBuilder<(ins "::llzk::function::FuncDefOp":$callee,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state, callee.getResultTypes(),
                          callee.getFullyQualifiedName(false),
                          argOperands, templateParams);
                  }]>,
       OpBuilder<(ins "::llzk::function::FuncDefOp":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state, callee.getResultTypes(),
                          callee.getFullyQualifiedName(false), mapOperands, numDimsPerMap,
                          argOperands, templateParams);
                  }]>,
       OpBuilder<(ins "::llzk::function::FuncDefOp":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state, callee, mapOperands,
                          $_builder.getDenseI32ArrayAttr(numDimsPerMap),
                          argOperands, templateParams);
                  }]>];

  let extraClassDeclaration = [{
    /// Required by CallOpInterface
    ::mlir::Operation *resolveCallableInTable(::mlir::SymbolTableCollection *symbolTable);

    /// Required by CallOpInterface
    ::mlir::Operation *resolveCallable();

    /// Return the FunctionType inferred from the arg operands and result types of this CallOp.
    /// This is not necessarily the same as the callee's FunctionType but should unify with it
    /// or else IR verification will fail.
    ::mlir::FunctionType getTypeSignature();

    /// Attempt type unfication between the inferred FunctionType from this CallOp (as LHS) and
    /// the given FunctionType (as RHS). If successful, return a UnificationMap containing the
    /// unifications that were made. Otherwise, return failure.
    ::mlir::FailureOr<UnificationMap> unifyTypeSignature(::mlir::FunctionType other);

    /// Return `true` iff the callee function name is `FUNC_NAME_COMPUTE` (this
    /// does not check if the callee function is located within a StructDefOp).
    inline bool calleeIsCompute() {
      return FUNC_NAME_COMPUTE == getCallee().getLeafReference();
    }

    /// Return `true` iff the callee function can contain witness generation code
    /// (this does not check if the callee function is located within a StructDefOp)
    inline bool calleeContainsWitnessGen() {
      return FUNC_NAME_COMPUTE == getCallee().getLeafReference() ||
             FUNC_NAME_PRODUCT == getCallee().getLeafReference();
    }

    /// Return `true` iff the callee function name is `FUNC_NAME_CONSTRAIN` (this
    /// does not check if the callee function is located within a StructDefOp).
    inline bool calleeIsConstrain() { return FUNC_NAME_CONSTRAIN == getCallee().getLeafReference(); }

    /// Return `true` iff the callee function name is `FUNC_NAME_COMPUTE` within a StructDefOp.
    bool calleeIsStructCompute();

    /// Return `true` iff the callee function name is `FUNC_NAME_CONSTRAIN` within a StructDefOp.
    bool calleeIsStructConstrain();

    /// Return the "self" value (i.e. the return value) from the callee function (which must be
    /// named `FUNC_NAME_COMPUTE`).
    ::mlir::Value getSelfValueFromCompute();

    /// Return the "self" value (i.e. the first parameter) from the callee function (which must be
    /// named `FUNC_NAME_CONSTRAIN`).
    ::mlir::Value getSelfValueFromConstrain();

    /// Resolve and return the target FuncDefOp for this CallOp.
    ::mlir::FailureOr<::llzk::SymbolLookupResult<::llzk::function::FuncDefOp>>
    getCalleeTarget(::mlir::SymbolTableCollection &tables);

    /// Assuming the callee is `FUNC_NAME_COMPUTE`, return the single StructType result.
    ::llzk::component::StructType getSingleResultTypeOfCompute();

    /// Assuming the callee contains witness generation code, return the single StructType result.
    ::llzk::component::StructType getSingleResultTypeOfWitnessGen();

    /// Allocate consecutive storage of the ValueRange instances in the parameter
    /// so it can be passed to the builders as an `ArrayRef<ValueRange>`.
    static ::llvm::SmallVector<::mlir::ValueRange> toVectorOfValueRange(::mlir::OperandRangeRange);

    /// Check type compatibility of the given template parameter value from this `CallOp` against
    /// the declared type on the given `TemplateParamOp` (if any).
    ::mlir::LogicalResult verifyTemplateParamCompatibility(
      ::mlir::Attribute paramFromCallOp, ::llzk::polymorphic::TemplateParamOp targetParam
    );

    /// Check type compatibility of each template parameter value provided in this `CallOp` against
    /// the declared type on each `TemplateParamOp` (if any).
    ///
    /// Pre-condition assertions:
    ///   - `!isNullOrEmpty(getTemplateParamsAttr())`
    ///   - `getTemplateParamsAttr().size() == llvm::range_size(targetParamDefs)`
    ::mlir::LogicalResult verifyTemplateParamCompatibility(
        ::llvm::iterator_range<::mlir::Region::op_iterator<::llzk::polymorphic::TemplateParamOp>> targetParamDefs
    );

    /// Verify that each template parameter value provided in this `CallOp` is consistent with
    /// the value inferred for the target `TemplateParamOp` in the given `UnificationMap`. The
    /// `UnificationMap` is expected to contain the unification results of this `CallOp` against
    /// the target function type signature.
    ///
    /// Pre-condition assertions:
    ///   - `!isNullOrEmpty(getTemplateParamsAttr())`
    ///   - `getTemplateParamsAttr().size() == llvm::range_size(targetParamDefs)`
    ::mlir::LogicalResult verifyTemplateParamsMatchInferred(
        ::llvm::iterator_range<::mlir::Region::op_iterator<::llzk::polymorphic::TemplateParamOp>> targetParamDefs,
        const UnificationMap &unifications
    );
  }];
}

#endif // LLZK_FUNC_OPS
