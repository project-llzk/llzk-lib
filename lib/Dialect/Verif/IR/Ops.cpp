//===-- Ops.cpp - Verif operation implementations ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/IR/Ops.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Twine.h>

// TableGen'd implementation files
#include "llzk/Dialect/Verif/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Verif/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::polymorphic;
using namespace llzk::felt;
using namespace llzk::component;
using namespace llzk::function;

namespace {

using namespace llzk::verif;

ParseResult parseContractOp(
    OpAsmParser &parser, OperationState &result, StringAttr typeAttrName,
    function_interface_impl::FuncTypeBuilder funcTypeBuilder, StringAttr argAttrsName
) {
  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  auto &builder = parser.getBuilder();

  // Parse the name as a symbol.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), result.attributes)) {
    return failure();
  }

  // Parse the target symbol
  StringAttr targetAttr;
  if (parser.parseKeyword("for") || parser.parseSymbolName(
                                        targetAttr,
                                        ContractOp::getTargetSymNameAttrName(OperationName(
                                            ContractOp::getOperationName(), parser.getContext()
                                        )),
                                        result.attributes
                                    )) {
    return failure();
  }

  // Parse the function signature.
  SMLoc signatureLocation = parser.getCurrentLocation();
  bool isVariadic = false;

  if (function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic*/ false, entryArgs, isVariadic, resultTypes, resultAttrs
      )) {
    return failure();
  }
  assert(isVariadic == false);
  // There should be no return types or attributes.
  if (!resultTypes.empty() || !resultAttrs.empty()) {
    return failure();
  }

  std::string errorMessage;
  SmallVector<Type> argTypes;
  argTypes.reserve(entryArgs.size());
  for (auto &arg : entryArgs) {
    argTypes.push_back(arg.type);
  }
  Type type = funcTypeBuilder(
      builder, argTypes, resultTypes, function_interface_impl::VariadicFlag(isVariadic),
      errorMessage
  );
  if (!type) {
    return parser.emitError(signatureLocation)
           << "failed to construct function type" << (errorMessage.empty() ? "" : ": ")
           << errorMessage;
  }
  result.addAttribute(typeAttrName, TypeAttr::get(type));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes)) {
    return failure();
  }

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed :
       {SymbolTable::getVisibilityAttrName(), SymbolTable::getSymbolAttrName(),
        typeAttrName.getValue()}) {
    if (parsedAttributes.get(disallowed)) {
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
    }
  }
  result.attributes.append(parsedAttributes);

  // Add the attributes to the function arguments.
  function_interface_impl::addArgAndResultAttrs(
      builder, result, entryArgs, resultAttrs, argAttrsName,
      /*resAttrsName*/ StringAttr::get(parser.getContext())
  );

  // Parse the required contract body.
  auto *body = result.addRegion();
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseRegion(
          *body, entryArgs,
          /*enableNameShadowing=*/false
      )) {
    return failure();
  }

  // Contract body was parsed, make sure its not empty.
  if (body->empty()) {
    return parser.emitError(loc, "expected non-empty contract body");
  }

  return success();
}

} // namespace

namespace llzk::verif {

//===------------------------------------------------------------------===//
// ContractOp
//===------------------------------------------------------------------===//

LogicalResult ContractOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Verify the target of the contract
  SymbolRefAttr targetSym = SymbolRefAttr::get(getTargetSymNameAttr());
  FailureOr<SymbolLookupResultUntyped> targetRes = lookupTopLevelSymbol(tables, targetSym, *this);
  if (failed(targetRes)) {
    return emitOpError().append("could not find target \"@", getTargetSymName(), "\"");
  }

  FunctionType contractTy = getFunctionType();
  // Verify the symbols in the contract argument
  if (failed(verifyTypeResolution(tables, *this, contractTy))) {
    // verifyTypeResolution already reports error messages
    return failure();
  }

  // Verify the target symbol
  Operation *targetOp = targetRes->get();
  if (StructDefOp structTarget = dyn_cast<StructDefOp>(targetOp)) {
    // Contract args must match constrain function arguments
    // TODO: product program support?
    if (!structTarget.hasComputeConstrain()) {
      return emitOpError().append(
          "contracts are not supported for \"@", FUNC_NAME_PRODUCT, "\"-style structs"
      );
    }
    FuncDefOp constrainFn = structTarget.getConstrainFuncOp();
    if (constrainFn.getFunctionType() != contractTy) {
      return emitOpError()
          .append(
              "contract type does not match struct \"@", FUNC_NAME_CONSTRAIN, "\" function type"
          )
          .attachNote(constrainFn.getLoc())
          .append("function defined here");
    }
  } else if (FuncDefOp funcTarget = dyn_cast<FuncDefOp>(targetOp)) {
    // Function args must match exactly
    if (funcTarget.getFunctionType() != contractTy) {
      return emitOpError()
          .append("contract type does not match function type")
          .attachNote(funcTarget.getLoc())
          .append("function defined here");
    }
  } else {
    // Disallowed
    return emitOpError().append("unsupported target type \"", targetOp->getName(), "\"");
  }

  // Also confirm that the types are in the same template op if they are.
  TemplateOp contractTmpl = getParentOfType<TemplateOp>(*this);
  TemplateOp targetTmpl = targetOp->getParentOfType<TemplateOp>();
  if (contractTmpl != targetTmpl) {
    if (targetTmpl) {
      return emitOpError()
          .append("contract must reside within the template containing the target")
          .attachNote(targetTmpl.getLoc())
          .append("target template defined here");
    } else if (contractTmpl) {
      return emitOpError()
          .append("contract cannot be within a template that does not contain the target")
          .attachNote(contractTmpl.getLoc())
          .append("contract template defined here");
    }
  }

  return success();
}

// Parse the ContractOp syntax using the built-in parsing of function-like
// operations. We'll verify contract-specific restrictions in `verify`.
ParseResult ContractOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return parseContractOp(
      parser, result, getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name)
  );
}

void ContractOp::print(OpAsmPrinter &p) {
  // Print the operation and the contract name.
  p << ' ';
  p.printSymbolName(getSymName());

  // Print the name of the contract's target.
  p << " for ";
  p.printSymbolName(getTargetSymName());
  p << ' ';

  ArrayRef<Type> argTypes = getArgumentTypes();
  function_interface_impl::printFunctionSignature(
      p, *this, argTypes, /*isVariadic*/ false, /*resultTypes*/ ArrayRef<Type>()
  );
  function_interface_impl::printFunctionAttributes(
      p, *this,
      /*elided*/ {getFunctionTypeAttrName(), getArgAttrsAttrName(), getTargetSymNameAttrName()}
  );
  // Print the body.
  Region &body = getRegion();
  p << ' ';
  p.printRegion(
      body, /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/true
  );
}

LogicalResult ContractOp::verify() {
  OwningEmitErrorFn emitErrorFunc = getEmitOpErrFn(this);

  if ((*this)->hasAttr(ARG_NAME_ATTR_NAME)) {
    return emitOpError() << "'" << ARG_NAME_ATTR_NAME << "' is only valid on function arguments";
  }

  if (ArrayAttr argAttrs = getAllArgAttrs()) {
    llvm::DenseSet<StringAttr> seenNames;
    for (auto [i, attr] : llvm::enumerate(argAttrs)) {
      auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attr);
      if (!dictAttr) {
        continue;
      }
      Attribute argNameAttr = dictAttr.get(ARG_NAME_ATTR_NAME);
      if (!argNameAttr) {
        continue;
      }
      auto argName = llvm::dyn_cast<StringAttr>(argNameAttr);
      if (!argName) {
        return emitOpError() << "'" << ARG_NAME_ATTR_NAME << "' on argument " << i
                             << " must be a string attribute";
      }
      if (!llvm::isa<NoneType>(argName.getType())) {
        return emitOpError() << "'" << ARG_NAME_ATTR_NAME << "' on argument " << i
                             << " must not have an explicit type";
      }
      if (argName.getValue().empty()) {
        return emitOpError() << "'" << ARG_NAME_ATTR_NAME << "' on argument " << i
                             << " must not be empty";
      }
      if (!seenNames.insert(argName).second) {
        return emitOpError() << "duplicate '" << ARG_NAME_ATTR_NAME << "' value \""
                             << argName.getValue() << "\" on argument " << i;
      }
    }
  }

  // Ensure that only valid LLZK types are used for contract arguments.
  FunctionType type = getFunctionType();
  for (Type t : type.getInputs()) {
    if (llzk::checkValidType(emitErrorFunc, t).failed()) {
      return failure();
    }
  }

  // Ensure that the contract does not contain nested modules, structs, or functions.
  WalkResult res = this->walk<WalkOrder::PreOrder>([this](Operation *op) {
    if (isa<ModuleOp, TemplateOp, FuncDefOp, StructDefOp>(op)) {
      getEmitOpErrFn(op)().append(
          "cannot be nested within '", getOperation()->getName(), "' operations"
      );
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (res.wasInterrupted()) {
    return failure();
  }

  return success();
}

//===------------------------------------------------------------------===//
// IncludeOp
//===------------------------------------------------------------------===//

void IncludeOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, SymbolRefAttr callee, ValueRange argOperands,
    ArrayRef<Attribute> templateParams
) {
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrsEmpty<IncludeOp>(
      odsBuilder, odsState, llzk::checkedCast<int32_t>(argOperands.size())
  );
  props.setCallee(callee);
  addTemplateParams<IncludeOp>(odsBuilder, props, templateParams);
}

void IncludeOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, SymbolRefAttr callee,
    ArrayRef<ValueRange> mapOperands, DenseI32ArrayAttr numDimsPerMap, ValueRange argOperands,
    ArrayRef<Attribute> templateParams
) {
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrs<IncludeOp>(
      odsBuilder, odsState, mapOperands, numDimsPerMap,
      llzk::checkedCast<int32_t>(argOperands.size())
  );
  props.setCallee(callee);
  addTemplateParams<IncludeOp>(odsBuilder, props, templateParams);
}

LogicalResult IncludeOp::verifyTemplateParamCompatibility(
    Attribute paramFromIncludeOp, TemplateParamOp targetParam
) {
  // A wildcard `?` (represented as kDynamic) defers inference to a later pass.
  // It is only valid for parameters with a `!poly.tvar` type restriction.
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(paramFromIncludeOp)) {
    if (isDynamic(intAttr)) {
      std::optional<Type> declaredType = targetParam.getTypeOpt();
      if (!declaredType || !llvm::isa<TypeVarType>(*declaredType)) {
        auto diag = this->emitOpError().append(
            "wildcard `?` can only be used for template parameters with `!poly.tvar` "
            "type restriction, but parameter \"@",
            targetParam.getName(), "\" has "
        );
        if (declaredType) {
          diag.append("type restriction ", *declaredType);
        } else {
          diag.append("no type restriction");
        }
        return diag;
      }
      return success();
    }
  }
  if (std::optional<Type> declaredType = targetParam.getTypeOpt()) {
    // Note: `declaredType` is restricted by `isValidConstReadType()`
    bool compatible = false;
    if (llvm::isa<TypeVarType>(*declaredType)) {
      compatible = llvm::isa<TypeAttr>(paramFromIncludeOp);
    } else if (llvm::isa<FeltType>(*declaredType)) {
      compatible = llvm::isa<FeltConstAttr, IntegerAttr>(paramFromIncludeOp) &&
                   isValidConstReadType(llvm::cast<TypedAttr>(paramFromIncludeOp).getType());
    } else if (llvm::isa<IndexType, IntegerType>(*declaredType)) {
      // Note: Just like struct type instantiation, there is no restriction on passing a
      // larger value to an `i1`. The flattening pass will treat 0 as false and any other
      // value as true (but give a warning if it's not 1).
      compatible = llvm::isa<IntegerAttr>(paramFromIncludeOp) &&
                   isValidConstReadType(llvm::cast<TypedAttr>(paramFromIncludeOp).getType());
    } else {
      llvm_unreachable("inconsistent with `isValidConstReadType()`");
    }
    if (!compatible) {
      // Tested in call_with_template_params_fail.llzk
      return this->emitOpError().append(
          "instantiation value '", paramFromIncludeOp, "' is not compatible with parameter \"@",
          targetParam.getName(), "\" type restriction ", *declaredType
      );
    }
  }
  return success();
}

LogicalResult IncludeOp::verifyTemplateParamCompatibility(
    llvm::iterator_range<Region::op_iterator<TemplateParamOp>> targetParamDefs
) {
  ArrayAttr callParams = this->getTemplateParamsAttr();
  assert(!isNullOrEmpty(callParams) && "pre-condition");
  assert((callParams.size() == llvm::range_size(targetParamDefs)) && "pre-condition");

  for (auto [paramOp, attr] : llvm::zip_equal(targetParamDefs, callParams.getValue())) {
    if (failed(verifyTemplateParamCompatibility(attr, paramOp))) {
      return failure();
    }
  }
  return success();
}

LogicalResult IncludeOp::verifyTemplateParamsMatchInferred(
    llvm::iterator_range<Region::op_iterator<TemplateParamOp>> targetParamDefs,
    const UnificationMap &unifications
) {
  ArrayAttr callParams = this->getTemplateParamsAttr();
  assert(!isNullOrEmpty(callParams) && "pre-condition");
  assert((callParams.size() == llvm::range_size(targetParamDefs)) && "pre-condition");

  for (auto [paramOp, attr] : llvm::zip_equal(targetParamDefs, callParams.getValue())) {
    // Skip wildcards (`?` / kDynamic) - their value will be resolved by a later inference pass.
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      if (isDynamic(intAttr)) {
        continue;
      }
    }
    auto it = unifications.find({FlatSymbolRefAttr::get(paramOp.getNameAttr()), Side::RHS});
    if (it != unifications.end() && !typeParamsUnify({attr}, {it->second})) {
      // Tested in call_with_template_params_fail.llzk
      return this->emitOpError().append(
          "template instantiation value '", attr, "' for parameter \"@", paramOp.getName(),
          "\" conflicts with value '", it->second, "' inferred from function type signature"
      );
    }
  }
  return success();
}

namespace {

struct IncludeOpVerifier {
  explicit IncludeOpVerifier(IncludeOp *c) : includeOp(c) {}
  virtual ~IncludeOpVerifier() = default;

  LogicalResult verify() {
    // Rather than immediately returning on failure, we check all verifier steps and aggregate to
    // provide as many errors are possible in a single verifier run.
    LogicalResult aggregateResult = success();
    if (failed(verifyInputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyTemplateParams())) {
      aggregateResult = failure();
    }
    return aggregateResult;
  }

protected:
  IncludeOp *includeOp;

  virtual LogicalResult verifyInputs() = 0;
  virtual LogicalResult verifyTemplateParams() = 0;

  LogicalResult verifyNoTemplateInstantiations() {
    if (!isNullOrEmpty(includeOp->getTemplateParamsAttr())) {
      // Tested in call_with_template_params_fail.llzk
      return includeOp->emitOpError().append(
          "can only have template instantiations when targeting a templated contract"
      );
    }
    return success();
  }
};

struct KnownTargetVerifier : public IncludeOpVerifier {
  KnownTargetVerifier(IncludeOp *c, SymbolLookupResult<ContractOp> &&tgtRes)
      : IncludeOpVerifier(c), tgt(*tgtRes), tgtType(tgt.getFunctionType()),
        includeSymNames(tgtRes.getNamespace()) {}

  LogicalResult verifyInputs() override {
    return verifyTypesMatch(includeOp->getArgOperands().getTypes(), tgtType.getInputs(), "operand");
  }

  LogicalResult verifyTemplateParams() override {
    auto tgtOp = tgt.getOperation();
    if (TemplateOp tgtOpParent = getParentOfType<TemplateOp>(tgtOp)) {
      // When the target function is a free function within a TemplateOp, the IncludeOp may have
      // template parameter instantiations that must be checked against the template parameters.
      // - If the function type signature references all template parameters, then the parameter
      //   instantiation list on the IncludeOp is optional, otherwise it's required.
      // - If present, the instantiation list must provide a value for every template parameter
      //   and the value must be type-compatible with the parameter's declared type (if any).
      // - If present, the instantiation list must result in a function type signature that can
      //   be unified with the IncludeOp's operand and result types.
      auto realParams = tgtOpParent.getConstOps<TemplateParamOp>();
      ArrayAttr callParams = includeOp->getTemplateParamsAttr();

      // When there is no instantiation list, just ensure that it's not required.
      if (isNullOrEmpty(callParams)) {
        llvm::SmallDenseSet<SymbolRefAttr> referencedInSignature;
        llzk::getSymbolsUsedIn(tgtType.getInputs(), referencedInSignature);
        llzk::getSymbolsUsedIn(tgtType.getResults(), referencedInSignature);

        bool allParamsReferenced = llvm::all_of(realParams, [&](TemplateParamOp p) {
          return referencedInSignature.contains(FlatSymbolRefAttr::get(p.getNameAttr()));
        });
        if (allParamsReferenced) {
          return success();
        }
        // Tested in call_with_template_params_fail.llzk
        return includeOp->emitOpError().append(
            "must provide template instantiation parameters when calling \"@", tgt.getSymName(),
            "\" because not all template parameters of \"@", tgtOpParent.getSymName(),
            "\" appear in the function type signature"
        );
      }

      // Ensure `forceIntAttrTypes()` was successful on the IncludeOp's template parameters.
      if (failed(llzk::forceIntAttrTypes(callParams.getValue(), [this] {
        return llzk::InFlightDiagnosticWrapper(this->includeOp->emitOpError());
      }))) {
        return failure();
      }

      // The instantiation list is present. Check it has exactly one entry per template param.
      size_t numTemplateParams = llvm::range_size(realParams);
      if (callParams.size() != numTemplateParams) {
        // Tested in call_with_template_params_fail.llzk
        return includeOp->emitOpError().append(
            "template instantiation has ", callParams.size(), " parameter(s) but \"@",
            tgtOpParent.getSymName(), "\" expects ", numTemplateParams, " template parameter(s)"
        );
      }

      // Check type compatibility of each provided value with the declared parameter type (if any).
      if (failed(includeOp->verifyTemplateParamCompatibility(realParams))) {
        return failure();
      }

      // Check that the provided instantiation values are consistent with what type unification
      // of the target function types against the call's operand and result types would determine.
      FailureOr<UnificationMap> unifyResult = includeOp->unifyTypeSignature(tgtType);
      assert(succeeded(unifyResult) && "already checked by `verifyInputs()` and `verifyOutputs()`");
      return includeOp->verifyTemplateParamsMatchInferred(realParams, unifyResult.value());
    } else {
      // Non-template functions cannot contain template parameter instantiations.
      return verifyNoTemplateInstantiations();
    }
  }

private:
  template <typename T>
  LogicalResult
  verifyTypesMatch(ValueTypeRange<T> includeOpTypes, ArrayRef<Type> tgtTypes, const char *aspect) {
    if (tgtTypes.size() != includeOpTypes.size()) {
      return includeOp->emitOpError()
          .append("incorrect number of ", aspect, "s for callee, expected ", tgtTypes.size())
          .attachNote(tgt.getLoc())
          .append("callee defined here");
    }
    for (unsigned i = 0, e = tgtTypes.size(); i != e; ++i) {
      if (!typesUnify(includeOpTypes[i], tgtTypes[i], includeSymNames)) {
        return includeOp->emitOpError().append(
            aspect, " type mismatch: expected type ", tgtTypes[i], ", but found ",
            includeOpTypes[i], " for ", aspect, " number ", i
        );
      }
    }
    return success();
  }

  ContractOp tgt;
  FunctionType tgtType;
  std::vector<llvm::StringRef> includeSymNames;
};

} // namespace

LogicalResult IncludeOp::verifySymbolUses(SymbolTableCollection &tables) {
  // First, verify symbol resolution in all input and output types.
  if (failed(verifyTypeResolution(tables, *this, getTypeSignature()))) {
    return failure(); // verifyTypeResolution() already emits a sufficient error message
  }

  // Check that the callee attribute was specified.
  SymbolRefAttr calleeAttr = getCalleeAttr();
  if (!calleeAttr) {
    return emitOpError("requires a 'callee' symbol reference attribute");
  }

  // If the callee references a parameter of the template where this call appears, perform
  // the subset of checks that can be done even though the target is unknown.
  if (calleeAttr.getNestedReferences().size() == 1) {
    if (TemplateOp parent = getParentOfType<TemplateOp>(*this)) {
      if (parent.hasConstNamed<TemplateParamOp>(calleeAttr.getRootReference())) {
        return this->emitError("expected parameterized callee to target a struct function")
            .append(
                " (i.e. \"@", FUNC_NAME_PRODUCT, "\", \"@", FUNC_NAME_COMPUTE, "\", or \"@",
                FUNC_NAME_CONSTRAIN, "\")"
            );
      }
    }
  }

  // Otherwise, callee must be specified via full path from the root module. Perform the full set of
  // checks against the known target function.
  auto tgtOpt = lookupTopLevelSymbol<ContractOp>(tables, calleeAttr, *this);
  if (failed(tgtOpt)) {
    return this->emitError() << "expected '" << ContractOp::getOperationName() << "' named \""
                             << calleeAttr << '"';
  }
  return KnownTargetVerifier(this, std::move(*tgtOpt)).verify();
}

FunctionType IncludeOp::getTypeSignature() {
  return FunctionType::get(getContext(), getArgOperands().getTypes(), /*results*/ {});
}

FailureOr<UnificationMap> IncludeOp::unifyTypeSignature(FunctionType other) {
  UnificationMap unifications;
  if (functionTypesUnify(getTypeSignature(), other, {}, &unifications)) {
    return unifications;
  } else {
    return failure();
  }
}

FailureOr<SymbolLookupResult<ContractOp>>
IncludeOp::getCalleeTarget(SymbolTableCollection &tables) {
  Operation *thisOp = this->getOperation();
  auto root = getRootModule(thisOp);
  assert(succeeded(root));
  return llzk::lookupSymbolIn<ContractOp>(tables, getCallee(), root->getOperation(), thisOp);
}

/// Return the callee of this operation.
CallInterfaceCallable IncludeOp::getCallableForCallee() { return getCalleeAttr(); }

/// Set the callee for this operation.
void IncludeOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(llvm::cast<SymbolRefAttr>(callee));
}

SmallVector<ValueRange> IncludeOp::toVectorOfValueRange(OperandRangeRange input) {
  llvm::SmallVector<ValueRange, 4> output;
  output.reserve(input.size());
  for (OperandRange r : input) {
    output.push_back(r);
  }
  return output;
}

Operation *IncludeOp::resolveCallableInTable(SymbolTableCollection *symbolTable) {
  FailureOr<SymbolLookupResult<ContractOp>> res =
      llzk::resolveCallable<ContractOp>(*symbolTable, *this);
  if (failed(res) || res->isManaged()) {
    // Cannot return pointer to a managed Operation since it would cause memory errors.
    return nullptr;
  }
  return res->get();
}

Operation *IncludeOp::resolveCallable() {
  SymbolTableCollection tables;
  return resolveCallableInTable(&tables);
}

} // namespace llzk::verif
