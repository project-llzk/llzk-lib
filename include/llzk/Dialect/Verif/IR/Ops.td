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
include "llzk/Dialect/SMT/IR/SMTTypes.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "llzk/Dialect/Shared/OpTraits.td"
include "llzk/Dialect/Shared/Types.td"
include "llzk/Dialect/Felt/IR/Types.td"

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class VerifDialectOp<string mnemonic, list<Trait> traits = []>
    : Op<VerifDialect, mnemonic, traits>;

class ConditionOp<string mnemonic, list<Trait> traits = []>
    : VerifDialectOp<mnemonic, traits#[ConditionOpInterface]> {

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

class ContractConditionOp<string mnemonic, list<Trait> traits = []>
    : ConditionOp<mnemonic, traits#[HasAncestor<"::llzk::verif::ContractOp">]>;

//===------------------------------------------------------------------===//
// Struct verification condition operations
//===------------------------------------------------------------------===//

def VerifAssertOp : ConditionOp<"assert", []> {
  let summary = "An assertion in the target of a contract, derived from one of "
                "the contract's specifications";
  let description = summary;
}

def VerifProveOp : ConditionOp<"prove", []> {
  let summary = "A proof obligation in the target of a contract, derived from"
                "one of the contract's specifications";
  let description = summary;
}

def VerifSMTProveOp : Op<VerifDialect, "smt_prove", []> {
  let arguments = (ins BoolType:$condition);
  let results = (outs);
  let summary = "A lowered proof obligation over an SMT expression";
  let assemblyFormat = [{ $condition attr-dict }];
}

//===------------------------------------------------------------------===//
// Precondition operations
//===------------------------------------------------------------------===//

class RequireOpBase<string mnemonic, list<Trait> traits = []>
    : ContractConditionOp<mnemonic, traits#[PreconditionOpInterface]> {
  let description = [{
    Base of an operation that encodes a precondition.

    Preconditions:
    - May not depend on values transitively derived from struct member reads
    or from function return values.
    - May not be present on the `llzk.main` entry struct.
  }];
}

def RequireComputeOp : RequireOpBase<"require_compute"> {
  let summary = "witness computation precondition";
}

def RequireConstrainOp : RequireOpBase<"require_constrain"> {
  let summary = "constraint generation precondition";
}

//===------------------------------------------------------------------===//
// Postcondition operations
//===------------------------------------------------------------------===//

class EnsureOpBase<string mnemonic, list<Trait> traits = []>
    : ContractConditionOp<mnemonic, traits#[PostconditionOpInterface]>;

def EnsureComputeOp : EnsureOpBase<"ensure_compute"> {
  let summary = "witness computation postcondition";
}

def EnsureConstrainOp : EnsureOpBase<"ensure_constrain"> {
  let summary = "constraint generation postcondition";
}

//===------------------------------------------------------------------===//
// ContractDefOp
//===------------------------------------------------------------------===//

def ContractOp
    : VerifDialectOp<
          "contract",
          [ParentOneOf<["::mlir::ModuleOp", "::llzk::polymorphic::TemplateOp"]>,
           DeclareOpInterfaceMethods<SymbolUserOpInterface>, AffineScope,
           AutomaticAllocationScope, FunctionOpInterface,
           SingleBlockImplicitTerminator<"::llzk::verif::ContractEndOp">,
           IsolatedFromAbove]> {
  let summary = "defines a specification contract for a given symbol";
  let description = [{
    A specification contract for a given function or struct in the circuit.
    Contracts are "function-like", as they have arguments matching their targets,
    and are "called" by the `verif.include` operation so that contracts can require
    other contracts to hold on subcomponents, etc.

    Contract argument rules:
    - For function targets, the contract accepts arguments matching the function's arguments.
    - For struct targets, the contract accepts arguments matching the argument list of the
    struct's `@constrain` function.

    Contracts do not return values. For function targets with return values, the
    return values are appended to the end of the contract's argument list.

    Contracts may not directly target struct functions; they must target the structs themselves.
    Contracts targeting the module's `llzk.main` struct may contain `verif.require_compute`
    or `verif.require_constrain` operations. A circuit whose main-level contract contains such
    preconditions is valid only for inputs that satisfy them.

    For templated targets (i.e., functions or structs within a `poly.template`), the contract
    body may reference the target template's parameters and expressions. If a contract is nested
    inside a `poly.template`, then the target must be in that same template.

    Examples:

    ```llzk

    struct.def @Bar {
      function.def @compute(%in : !felt.type) -> !struct.type<@Bar> { ... }
      function.def @constrain(%self : !struct.type<@Bar>, %in : !felt.type) { ... }
    }

    verif.contract @FooContract for @Bar (%self : !struct.type<@Bar>, %in : !felt.type) {
      ...
    }

    function.def @free_func (%a : !felt.type) -> !felt.type { ... }

    verif.contract @BarContract for @free_func (%a : !felt.type, %ret_value : !felt.type) {
      ...
    }

    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, SymbolRefAttr:$target,
      TypeAttrOf<FunctionType>:$function_type,
      OptionalAttr<DictArrayAttr>:$arg_attrs);

  let regions = (region AnyRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;

  // Define builders manually so builder-created contracts always contain a
  // complete single-block body with the implicit `verif.contract_end`.
  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::mlir::StringAttr":$sym_name,
                                "::mlir::SymbolRefAttr":$target,
                                "::mlir::TypeAttr":$function_type,
                                CArg<"::mlir::ArrayAttr", "{}">:$arg_attrs),
                            [{
        build($_builder, $_state, ::mlir::TypeRange {}, sym_name, target, function_type, arg_attrs);
      }]>,
                  OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                                "::mlir::StringAttr":$sym_name,
                                "::mlir::SymbolRefAttr":$target,
                                "::mlir::TypeAttr":$function_type,
                                CArg<"::mlir::ArrayAttr", "{}">:$arg_attrs),
                            [{
        $_state.getOrAddProperties<Properties>().sym_name = sym_name;
        $_state.getOrAddProperties<Properties>().target = target;
        $_state.getOrAddProperties<Properties>().function_type = function_type;
        if (arg_attrs) {
          $_state.getOrAddProperties<Properties>().arg_attrs = arg_attrs;
        }
        initializeEmptyBody(
            $_builder, $_state, ::mlir::cast<::mlir::FunctionType>(function_type.getValue())
        );
        assert(resultTypes.size() == 0u && "mismatched number of results");
        $_state.addTypes(resultTypes);
      }]>,
                  OpBuilder<(ins "::llvm::StringRef":$sym_name,
                                "::mlir::SymbolRefAttr":$target,
                                "::mlir::FunctionType":$function_type,
                                CArg<"::mlir::ArrayAttr", "{}">:$arg_attrs),
                            [{
        build($_builder, $_state, ::mlir::TypeRange {}, sym_name, target, function_type, arg_attrs);
      }]>,
                  OpBuilder<(ins "::mlir::TypeRange":$resultTypes,
                                "::llvm::StringRef":$sym_name,
                                "::mlir::SymbolRefAttr":$target,
                                "::mlir::FunctionType":$function_type,
                                CArg<"::mlir::ArrayAttr", "{}">:$arg_attrs),
                            [{
        $_state.getOrAddProperties<Properties>().sym_name = $_builder.getStringAttr(sym_name);
        $_state.getOrAddProperties<Properties>().target = target;
        $_state.getOrAddProperties<Properties>().function_type = ::mlir::TypeAttr::get(function_type);
        if (arg_attrs) {
          $_state.getOrAddProperties<Properties>().arg_attrs = arg_attrs;
        }
        initializeEmptyBody($_builder, $_state, function_type);
        assert(resultTypes.size() == 0u && "mismatched number of results");
        $_state.addTypes(resultTypes);
      }]>,
                  OpBuilder<(ins "::llvm::StringRef":$name,
                      "llvm::StringRef":$target)>,
                  OpBuilder<(ins "::llvm::StringRef":$name,
                      "::mlir::SymbolRefAttr":$target)>];

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
    bool hasStructTarget() { return succeeded(getStructTarget()); }

    /// Return the StructDefOp that this contract targets, or failure if it does not
    /// target a struct or the struct is not found.
    ::mlir::FailureOr<SymbolLookupResult<component::StructDefOp>> getStructTarget(::mlir::SymbolTableCollection &tables);

    ::mlir::FailureOr<SymbolLookupResult<component::StructDefOp>> getStructTarget() {
      ::mlir::SymbolTableCollection tables;
      return getStructTarget(tables);
    }

    /// Return the "self" value (i.e. the first parameter) from the contract, or
    /// failure if the contract does not target a struct.
    ::mlir::FailureOr<::mlir::Value> getSelfValue();

    /// Return `true` iff the contract targets a function.
    bool hasFuncTarget() { return succeeded(getFuncTarget()); }

    /// Return the FuncDefOp that this contract targets, or failure if it does not
    /// target a function or the function is not found.
    ::mlir::FailureOr<SymbolLookupResult<function::FuncDefOp>> getFuncTarget(::mlir::SymbolTableCollection &tables);

    ::mlir::FailureOr<SymbolLookupResult<function::FuncDefOp>> getFuncTarget() {
      ::mlir::SymbolTableCollection tables;
      return getFuncTarget(tables);
    }

    /// Return the operation that this contract targets, or failure if it does not
    /// target an operation that implements the `ContractTarget` op interface or is not found.
    ::mlir::FailureOr<SymbolLookupResult<::llzk::verif::ContractTargetOpInterface>> getTargetOp(::mlir::SymbolTableCollection &tables);

    ::mlir::FailureOr<SymbolLookupResult<::llzk::verif::ContractTargetOpInterface>> getTargetOp() {
      ::mlir::SymbolTableCollection tables;
      return getTargetOp(tables);
    }

  private:
    /// Populate a builder-created contract body with one entry block matching
    /// the function signature and insert the implicit `verif.contract_end`.
    static void initializeEmptyBody(
        ::mlir::OpBuilder &builder, ::mlir::OperationState &state,
        ::mlir::FunctionType functionType
    );
  }];
}

//===------------------------------------------------------------------===//
// ContractEndOp
//===------------------------------------------------------------------===//

def ContractEndOp
    : VerifDialectOp<"contract_end", [Pure, Terminator,
                                      HasParent<"::llzk::verif::ContractOp">]> {
  let summary = "terminates a verif.contract body";
  let description = [{
    Implicit terminator for `verif.contract` regions. This operation is inserted
    automatically by the parser and omitted by the custom printer so contract
    syntax remains terminator-free.

    A terminator is expected by certain MLIR utilities (e.g., the data-flow analysis
    framework), which is why this op is added.
  }];

  let assemblyFormat = "attr-dict";
}

//===------------------------------------------------------------------===//
// IncludeOp
//===------------------------------------------------------------------===//

def IncludeOp
    : VerifDialectOp<
          "include", [MemRefsNormalizable, AttrSizedOperandSegments,
                      VerifySizesForMultiAffineOps<1>,
                      DeclareOpInterfaceMethods<CallOpInterface>,
                      DeclareOpInterfaceMethods<SymbolUserOpInterface>,
                      HasAncestor<"::llzk::verif::ContractOp">]> {
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
       OpBuilder<(ins "::llzk::verif::ContractOp":$callee,
                     CArg<"::mlir::ValueRange", "{}">:$argOperands,
                     CArg<"::llvm::ArrayRef<::mlir::Attribute>",
                          "{}">:$templateParams),
                 [{
                    build($_builder, $_state,
                          callee.getFullyQualifiedName(false),
                          argOperands, templateParams);
                  }]>,
       OpBuilder<(ins "::llzk::verif::ContractOp":$callee,
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
       OpBuilder<(ins "::llzk::verif::ContractOp":$callee,
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

//===------------------------------------------------------------------===//
// InvariantOp
//===------------------------------------------------------------------===//

def InvariantOp
    : VerifDialectOp<"invariant", [HasAncestor<"::llzk::verif::ContractOp">,
                                   NoTerminator, SingleBlock]> {

  let summary = "loop invariant definition operation";
  let description = [{
    Defines an invariant for a loop inside the target of a contract.

    The targeted loop op must implement the `InvariantTargetOpInterface`.
    This interface is already implemented by the `scf.while` and `scf.for`
    operations. The loop name is defined on either of those ops with a
    string attribute named `loop_label`.

    The arguments of the body must match those declared by the target op.
    In the case of `scf.for` the control values must also be block arguments
    in the following order: lower bound, induction variable, upper bound, and stride.

    Example:
    ```
    module attributes {llzk.lang} {
      struct.def @Top {
        function.def @compute() -> !struct.type<@Top> {
          %self = struct.new : !struct.type<@Top>
          %c0 = arith.constant 0 : index
          %c10 = arith.constant 10 : index
          %c1 = arith.constant 1  : index
          %0 = felt.const  5
          %1 = scf.for %iv = %c0 to %c10 step %c1 iter_args(%2 = %0) -> !felt.type {
            scf.yield %2 : !felt.type
          } {loop_label = "loopA"}
          function.return %self : !struct.type<@Top>
        }
        function.def @constrain(%self: !struct.type<@Top>)  {
          function.return
        }
      }

      verif.contract @Foo for @Top (%self: !struct.type<@Top>) {
        verif.invariant for @loopA(%lb: index, %iv: index, %ub: index, %step: index, %extra: !felt.type) {
          %true = arith.constant  true
          verif.require_compute %true
        }
      }
    }
    ```
  }];

  let arguments = (ins StrAttr:$loop_name, TypeArrayAttr:$loop_arg_types);

  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "::mlir::StringRef":$loop_name,
      CArg<"::llvm::ArrayRef<::mlir::Type>", "{}">:$loop_arg_types,
      CArg<"::llvm::ArrayRef<::mlir::Location>", "{}">:$loop_arg_locs)>];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the contract operation that contains this invariant.
    ::llzk::verif::ContractOp getParentContract();
    /// Returns the loop target.
    ::mlir::FailureOr<::llzk::verif::InvariantTargetOpInterface> getTarget();
  }];
}

//===------------------------------------------------------------------===//
// Invariant inner ops
//===------------------------------------------------------------------===//

class InvariantInnerOp<string mnemonic, list<Trait> traits = []>
    : VerifDialectOp<
          mnemonic,
          traits#[HasAncestor<"::llzk::verif::InvariantOp">,
                  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let summary = mnemonic#" value operation";
  let description = [{
    Indicates that a value }]#mnemonic#[{ on each iteration of the loop.

    This declares the `MemoryEffectsOpInterface`, which, like the `cf.assert` (MLIR `cf` dialect)
    and `bool.assert` (LLZK `bool` dialect) ops, adds a MemWrite affect to model program termination.
  }];

  let arguments = (ins LLZK_FeltType:$value);

  let extraClassDefinition = [{
    // This side effect models "program termination". Based on
    // https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
    void $cppClass::getEffects(
        ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects
    ) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get());
    }
  }];

  let assemblyFormat = "$value attr-dict";
}

def IncreasesOp : InvariantInnerOp<"increases"> {}

def DecreasesOp : InvariantInnerOp<"decreases"> {}

def StepOp : VerifDialectOp<
                 "step", [HasAncestor<"::llzk::verif::InvariantOp">,
                          DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
                          NoRegionArguments, SingleBlock]> {
  let summary = "step predicate operation";
  let description = [{
    Defines a predicate that must be satisfied between iterations of the loop.

    The predicate can access the value in the previous iteration by using the `verif.old` operation.

    Declares the `MemoryEffectsOpInterface`, which, like the `cf.assert` (MLIR `cf` dialect)
    and `bool.assert` (LLZK `bool` dialect) ops, adds a MemWrite affect to model program termination.
  }];

  let regions = (region SizedRegion<1>:$region);

  let extraClassDefinition = [{
    // This side effect models "program termination". Based on
    // https://github.com/llvm/llvm-project/blob/f325e4b2d836d6e65a4d0cf3efc6b0996ccf3765/mlir/lib/Dialect/ControlFlow/IR/ControlFlowOps.cpp#L92-L97
    void $cppClass::getEffects(
        ::mlir::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects
    ) {
      effects.emplace_back(::mlir::MemoryEffects::Write::get());
    }
  }];

  let assemblyFormat = "$region attr-dict";
}

def StepYieldOp
    : VerifDialectOp<"step.yield", [HasParent<"StepOp">, Terminator]> {
  let summary = "step yield operation";
  let description = [{
    Terminator operation for `verif.step` blocks. Takes the boolean value carrying the predicate's result.
  }];

  let arguments = (ins I1:$value);

  let assemblyFormat = "$value attr-dict";
}

def OldOp : VerifDialectOp<"old", [Pure, HasAncestor<"::llzk::verif::StepOp">,
                                   AllTypesMatch<["value", "result"]>]> {
  let summary = "old operation";
  let description = [{
    This operation allows accessing the value of an expression in the previous iteration of the loop.

    In can only be used within the body of a `verif.step` operation.
  }];

  let arguments = (ins AnyLLZKType:$value);
  let results = (outs AnyLLZKType:$result);

  let assemblyFormat = "$value `:` type($result) attr-dict";
}

#endif // LLZK_VERIF_OPS
