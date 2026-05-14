//===-- Ops.td ---------------------------------------------*- tablegen -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef LLZK_VERIF_OPS
#define LLZK_VERIF_OPS

include "llzk/Dialect/Verif/IR/Dialect.td"
include "llzk/Dialect/Verif/IR/OpInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "llzk/Dialect/Shared/OpTraits.td"
include "llzk/Dialect/Shared/Types.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class VerifDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<VerifDialect, mnemonic, traits>;

class ConditionOp<string mnemonic, list<Trait> traits = []>
    : VerifDialectOp<mnemonic,
                     traits#[ConditionOpInterface,
                             HasAncestor<"::llzk::verif::ContractOp">]> {

  let arguments = (ins I1:$condition);
  let results = (outs);

  let assemblyFormat = [{ $condition attr-dict }];

  let extraClassDefinition = [{
    // This side effect models "program termination". Based on
    // https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
    void $cppClass::getEffects(
        ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects
    ) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get());
    }
  }];
}

//===------------------------------------------------------------------===//
// Precondition operations
//===------------------------------------------------------------------===//

class RequireOpBase<string mnemonic, list<Trait> traits = []>
    : ConditionOp<mnemonic, traits#[PreconditionOpInterface]>;

def RequireComputeOp : RequireOpBase<"require_compute"> {
  let summary = "";
  let description = [{
  }];
}

def RequireConstrainOp : RequireOpBase<"require_constrain"> {
  let summary = "";
  let description = [{
  }];
}

//===------------------------------------------------------------------===//
// Postcondition operations
//===------------------------------------------------------------------===//

class EnsureOpBase<string mnemonic, list<Trait> traits = []>
    : ConditionOp<mnemonic, traits#[PostconditionOpInterface]>;

def EnsureComputeOp : EnsureOpBase<"ensure_compute"> {
  let summary = "";
  let description = [{
  }];
}

def EnsureConstrainOp : EnsureOpBase<"ensure_constrain"> {
  let summary = "";
  let description = [{
  }];
}

//===------------------------------------------------------------------===//
// ContractDefOp
//===------------------------------------------------------------------===//

// TODO: Build methods directly from FuncDefOp and StructDefOp

def ContractOp
    : VerifDialectOp<
          "contract", [ParentOneOf<["::mlir::ModuleOp",
                                    "::llzk::polymorphic::TemplateOp"]>,
                       DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                       AffineScope, AutomaticAllocationScope,
                       FunctionOpInterface, NoTerminator, IsolatedFromAbove]> {
  let summary = "defines a specification contract for a given symbol";
  let description = [{

    Example:

    verif.contract @FooContract for @Bar () {

    }
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, SymbolRefAttr:$target,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs);

  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [OpBuilder<(ins "::llvm::StringRef":$name,
      "llvm::StringRef":$target)>];

  let extraClassDeclaration = [{
    /// Create a deep copy of this contract and all of its blocks, remapping any
    /// operands that use values outside of the contract using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for contract arguments, these arguments are not
    /// included in the new contract. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    ContractOp clone(::mlir::IRMapping &mapper);
    ContractOp clone();

    /// Clone the internal blocks and attributes from this contract into dest.
    /// Any cloned blocks are appended to the back of dest. This contract
    /// asserts that the attributes of the current contract and dest are
    /// compatible.
    void cloneInto(ContractOp dest, ::mlir::IRMapping &mapper);

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
    /// Returns the region on the current operation that is callable.
    ::mlir::Region *getCallableRegion() { return &getBody(); }

    /// Required by FunctionOpInterface.
    /// Returns the argument types of this contract.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Required by FunctionOpInterface.
    /// Returns the result types of this contract. Since contracts don't have
    /// return values, the returned ArrayRef will always be empty.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Required by SymbolOpInterface.
    bool isDeclaration() { return false; }

    /// Return the full name for this contract from the root module, including
    /// all surrounding symbol table names (i.e., modules and structs).
    ::mlir::SymbolRefAttr getFullyQualifiedName(bool requireParent = true);

    /// Return `true` iff the contract targets a struct type.
    bool hasStructTarget();

    /// Return the "self" value (i.e. the first parameter) from the contract (the contract
    /// must target a struct).
    ::mlir::Value getSelfValue();
  }];
}

//===------------------------------------------------------------------===//
// IncludeOp
//===------------------------------------------------------------------===//

def IncludeOp
    : VerifDialectOp<
          "include", [MemRefsNormalizable, AttrSizedOperandSegments,
                      VerifySizesForMultiAffineOps<1>,
                      DeclareOpInterfaceMethods<CallOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "contract inclusion operation";
  let description = [{
    Invokes another specification contract from another contract, effectively including the
    specifications from another specification into the current contract.
  }];

  // See `VerifySizesForMultiAffineOps` for more explanation of these arguments.
  let arguments = (ins
      // Call target contract reference.
      SymbolRefAttr:$callee,
      // List of arguments to call the target contract.
      Variadic<AnyLLZKType>:$argOperands,
      // List of parameters to instantiate all `poly.param` symbols when the
      // callee is a contract inside a `poly.template` region.
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

  let assemblyFormat = [{
    $callee
    ( `<` custom<TemplateParams>($templateParams)^ `>` )?
    `` `(` $argOperands `)`
    ( `{` custom<MultiDimAndSymbolList>($mapOperands, $numDimsPerMap)^ `}` )?
    `:` functional-type($argOperands, results)
    custom<AttrDictWithWarnings>(attr-dict, prop-dict)
  }];

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
      [OpBuilder<(ins "::mlir::SymbolRefAttr":$callee,
           CArg<"::mlir::ValueRange", "{}">:$argOperands,
           CArg<"::llvm::ArrayRef<::mlir::Attribute>", "{}">:$templateParams)>,
       OpBuilder<(ins "::mlir::SymbolRefAttr":$callee,
           "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
           "::mlir::DenseI32ArrayAttr":$numDimsPerMap,
           CArg<"::mlir::ValueRange", "{}">:$argOperands,
           CArg<"::llvm::ArrayRef<::mlir::Attribute>", "{}">:$templateParams)>,
       OpBuilder<(ins "::mlir::SymbolRefAttr":$callee,
                     "::llvm::ArrayRef<::mlir::ValueRange>":$mapOperands,
                     "::llvm::ArrayRef<int32_t>":$numDimsPerMap,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state, callee, mapOperands,
                          $_builder.getDenseI32ArrayAttr(numDimsPerMap),
                          argOperands, templateParams);
                  }]>,
       OpBuilder<(ins "::llzk::function::FuncDefOp":$callee,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state,
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
                    build($_builder, $_state,
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

    /// Return the FunctionType inferred from the arg operands of this CallOp.
    /// This is not necessarily the same as the callee's FunctionType but should unify with it
    /// or else IR verification will fail.
    ::mlir::FunctionType getTypeSignature();

    /// Attempt type unfication between the inferred FunctionType from this CallOp (as LHS) and
    /// the given FunctionType (as RHS). If successful, return a UnificationMap containing the
    /// unifications that were made. Otherwise, return failure.
    ::mlir::FailureOr<UnificationMap> unifyTypeSignature(::mlir::FunctionType other);

    /// Return `true` iff the contract targets a struct type.
    bool contractTargetsStruct();

    /// Return the "self" value (i.e. the first parameter) from the callee contract,
    /// assuming the target of the contract is a struct target.
    ::mlir::Value getSelfValue();

    /// Resolve and return the target Contract for this CallOp.
    ::mlir::FailureOr<::llzk::SymbolLookupResult<::llzk::verif::ContractOp>>
    getCalleeTarget(::mlir::SymbolTableCollection &tables);

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

#endif // LLZK_VERIF_OPS
