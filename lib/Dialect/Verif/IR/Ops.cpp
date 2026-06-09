//===-- Ops.cpp - Verif operation implementations ---------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/IR/Ops.h"

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Analysis/SourceRef.h"
#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Dialect/Verif/Util/ForbiddenPreconditionInfluence.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
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

// Check if the op is a valid contract target.
bool isValidTarget(Operation *op) {
  if (auto fnOp = dyn_cast<FuncDefOp>(op)) {
    // Cannot target struct functions directly
    return fnOp->getParentOfType<StructDefOp>() == nullptr;
  }
  // Only other supported target currently is a struct
  return isa<StructDefOp>(op);
}

struct TargetTypeInfo {
  FunctionType funcType {};
  ArrayAttr argAttrs {};
};

FailureOr<TargetTypeInfo> getTargetTypeInfo(Operation *op) {
  if (auto fnOp = dyn_cast<FuncDefOp>(op)) {
    // Recreate the function type with return types appended to the arguments.
    FunctionType fnTy = fnOp.getFunctionType();
    ArrayRef<Type> curInputs = fnTy.getInputs(), curResults = fnTy.getResults();
    SmallVector<Type> newInputs;
    newInputs.reserve(curInputs.size() + curResults.size());
    newInputs.insert(newInputs.end(), curInputs.begin(), curInputs.end());
    newInputs.insert(newInputs.end(), curResults.begin(), curResults.end());
    // And no return types
    auto newFnTy = fnTy.clone(newInputs, {});
    // Do the same appending to the return attrs
    ArrayAttr curArgAttrs = fnOp.getArgAttrsAttr(), curResAttrs = fnOp.getResAttrsAttr();
    ArrayAttr newArgAttrsAttr {};
    if (curArgAttrs || curResAttrs) {
      auto *ctx = op->getContext();
      SmallVector<Attribute> newArgAttrs;
      // Since there are some attributes, it must match the length of the input arguments
      newArgAttrs.reserve(newInputs.size());
      if (curArgAttrs) {
        newArgAttrs.insert(newArgAttrs.end(), curArgAttrs.begin(), curArgAttrs.end());
      } else {
        // Pad
        newArgAttrs.insert(newArgAttrs.end(), curInputs.size(), DictionaryAttr::get(ctx));
      }
      if (curResAttrs) {
        newArgAttrs.insert(newArgAttrs.end(), curResAttrs.begin(), curResAttrs.end());
      } else {
        // pad
        newArgAttrs.insert(newArgAttrs.end(), curResults.size(), DictionaryAttr::get(ctx));
      }
      newArgAttrsAttr = ArrayAttr::get(ctx, newArgAttrs);
    }

    return TargetTypeInfo {
        .funcType = newFnTy,
        .argAttrs = newArgAttrsAttr,
    };
  }
  if (auto structOp = dyn_cast<StructDefOp>(op)) {
    if (structOp.hasComputeConstrain()) {
      auto fnOp = structOp.getConstrainFuncOp();
      return TargetTypeInfo {
          .funcType = fnOp.getFunctionType(),
          .argAttrs = fnOp.getArgAttrsAttr(),
      };
    } else {
      FuncDefOp productFn = structOp.getProductFuncOp();
      assert(productFn);
      // Augment the product function signature to accept the self argument.
      FunctionType fnTy = productFn.getFunctionType();
      ArrayRef<Type> curInputs = fnTy.getInputs();
      // Accept the self struct type in addition to existing inputs
      SmallVector<Type> newInputs;
      newInputs.reserve(curInputs.size() + 1);
      newInputs.push_back(structOp.getType());
      newInputs.insert(newInputs.end(), curInputs.begin(), curInputs.end());
      // And no return types
      auto newFnTy = fnTy.clone(newInputs, {});
      // We also need to expand the arg attributes by one
      auto *ctx = op->getContext();
      ArrayAttr curArgAttrs = productFn.getArgAttrsAttr();
      ArrayAttr newArgAttrsAttr = curArgAttrs;
      if (curArgAttrs) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(curArgAttrs.size() + 1);
        newArgAttrs.push_back(DictionaryAttr::get(ctx));
        newArgAttrs.insert(newArgAttrs.end(), curArgAttrs.begin(), curArgAttrs.end());
        newArgAttrsAttr = ArrayAttr::get(ctx, newArgAttrs);
      }
      return TargetTypeInfo {
          .funcType = newFnTy,
          .argAttrs = newArgAttrsAttr,
      };
    }
  }

  return failure();
}

enum class ForbiddenRequireConditionKind : uint8_t {
  MainContract,
  StructMember,
  FunctionReturn,
};

std::optional<ForbiddenRequireConditionKind>
classifyForbiddenConditionProvenance(ModuleOp module, Value value, ContractOp contract) {
  ForbiddenPreconditionInfluence influence =
      analyzeForbiddenPreconditionInfluence(module, contract, value);
  if (hasInfluence(influence, ForbiddenPreconditionInfluence::StructMember)) {
    return ForbiddenRequireConditionKind::StructMember;
  }
  if (hasInfluence(influence, ForbiddenPreconditionInfluence::FunctionReturn)) {
    return ForbiddenRequireConditionKind::FunctionReturn;
  }
  return std::nullopt;
}

// Map a classified restriction failure to the verifier diagnostic emitted on
// the offending require op.
LogicalResult
emitForbiddenRequireCondition(Operation *requireOp, ForbiddenRequireConditionKind kind) {
  switch (kind) {
  case ForbiddenRequireConditionKind::MainContract:
    return requireOp->emitOpError(
        "cannot appear directly in a contract that targets the main entry-point struct"
    );
  case ForbiddenRequireConditionKind::StructMember:
    return requireOp->emitOpError("condition cannot be derived from a struct member value");
  case ForbiddenRequireConditionKind::FunctionReturn:
    return requireOp->emitOpError("condition cannot be derived from a function return value");
  }
  llvm_unreachable("unknown forbidden require condition kind");
}

// Enforce the direct main-contract ban and the provenance restrictions on
// require ops that appear in this contract body.
LogicalResult verifyRequireRestrictions(ContractOp contract) {
  SmallVector<Operation *> requireOps;
  contract.walk([&](Operation *op) {
    if (isa<RequireComputeOp, RequireConstrainOp>(op)) {
      requireOps.push_back(op);
    }
  });
  if (requireOps.empty()) {
    return success();
  }

  bool targetsMainStruct = false;
  {
    SymbolTableCollection tables;
    auto structTarget = contract.getStructTarget(tables);
    targetsMainStruct = succeeded(structTarget) && structTarget->get().isMainComponent();
  }

  for (Operation *requireOp : requireOps) {
    if (targetsMainStruct) {
      return emitForbiddenRequireCondition(requireOp, ForbiddenRequireConditionKind::MainContract);
    }
  }

  ModuleOp module = contract->getParentOfType<ModuleOp>();
  if (!module) {
    return contract.emitOpError("must have a parent module to analyze condition provenance");
  }

  for (Operation *requireOp : requireOps) {
    Value condition = requireOp->getOperand(0);
    if (auto kind = classifyForbiddenConditionProvenance(module, condition, contract)) {
      return emitForbiddenRequireCondition(requireOp, *kind);
    }
  }

  return success();
}

} // namespace

namespace llzk::verif {

//===------------------------------------------------------------------===//
// ContractOp
//===------------------------------------------------------------------===//

void ContractOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name,
    llvm::StringRef target
) {
  build(odsBuilder, odsState, name, SymbolRefAttr::get(odsBuilder.getContext(), target));
}

void ContractOp::build(
    ::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::llvm::StringRef name,
    ::mlir::SymbolRefAttr target
) {
  // Any errors here in the construction from the target information are not
  // reported here, but will instead be reported when the verify function fails
  // to verify this op.
  SymbolTableCollection tables;
  // Find the target of the contract
  FailureOr<SymbolLookupResultUntyped> targetRes =
      lookupTopLevelSymbol(tables, target, odsBuilder.getBlock()->getParentOp());
  if (failed(targetRes)) {
    return;
  }
  Operation *targetOp = targetRes->get();
  if (!isValidTarget(targetOp)) {
    return;
  }
  FailureOr<TargetTypeInfo> infoRes = getTargetTypeInfo(targetOp);
  if (failed(infoRes)) {
    return;
  }
  TargetTypeInfo &info = *infoRes;
  build(odsBuilder, odsState, name, target, info.funcType, info.argAttrs);
}

bool ContractOp::hasArgPublicAttr(unsigned index) {
  if (index < this->getNumArguments()) {
    DictionaryAttr res = function_interface_impl::getArgAttrDict(*this, index);
    return res ? res.contains(PublicAttr::name) : false;
  }
  return false;
}

bool ContractOp::hasArgName(unsigned index) { return static_cast<bool>(getArgNameAttr(index)); }

std::optional<StringAttr> ContractOp::getArgNameAttr(unsigned index) {
  if (index >= getNumArguments()) {
    return std::nullopt;
  }
  if (StringAttr attr = getArgAttrOfType<StringAttr>(index, ARG_NAME_ATTR_NAME)) {
    return attr;
  }
  return std::nullopt;
}

void ContractOp::setArgNameAttr(unsigned index, const StringAttr &attr) {
  assert(index < getNumArguments() && "argument index out of range");
  setArgAttr(index, ARG_NAME_ATTR_NAME, attr);
}

void ContractOp::setArgName(unsigned index, llvm::StringRef name) {
  setArgNameAttr(index, StringAttr::get(getContext(), name));
}

SymbolRefAttr ContractOp::getFullyQualifiedName(bool requireParent) {
  if (!requireParent && getOperation()->getParentOp() == nullptr) {
    return SymbolRefAttr::get(getOperation());
  }
  auto res = getPathFromRoot(*this);
  assert(succeeded(res));
  return res.value();
}

LogicalResult ContractOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Verify the target of the contract
  FailureOr<ModuleOp> rootRes = getRootModule(getOperation());
  if (failed(rootRes)) {
    return emitOpError().append("could not lookup root module");
  }
  FailureOr<SymbolLookupResultUntyped> targetRes =
      lookupTopLevelSymbol(tables, getTargetAttr(), rootRes->getOperation());
  if (failed(targetRes)) {
    return emitOpError().append("could not find target \"@", getTarget(), "\"");
  }

  FunctionType contractTy = getFunctionType();
  // Verify the symbols in the contract argument
  if (failed(verifyTypeResolution(tables, *this, contractTy))) {
    // verifyTypeResolution already reports error messages
    return failure();
  }

  // Verify the target symbol
  Operation *targetOp = targetRes->get();
  if (!isValidTarget(targetOp)) {
    return emitOpError()
        .append("target \"", getTargetAttr(), "\" is not a supported contract target")
        .attachNote(targetOp->getLoc())
        .append("target defined here");
  }
  FailureOr<TargetTypeInfo> targetInfoRes = getTargetTypeInfo(targetOp);
  if (failed(targetInfoRes)) {
    return emitOpError()
        .append("unsupported target type \"", targetOp->getName(), "\"")
        .attachNote(targetOp->getLoc())
        .append("target defined here");
  }
  TargetTypeInfo &targetInfo = *targetInfoRes;
  if (targetInfo.funcType != contractTy) {
    return emitOpError()
        .append("contract type does not match target type")
        .attachNote(targetOp->getLoc())
        .append("target defined here");
  }
  if (targetInfo.argAttrs != getArgAttrsAttr()) {
    return emitOpError()
        .append(
            "contract arg attributes ", getArgAttrsAttr(), " does not match target arg attributes ",
            targetInfo.argAttrs
        )
        .attachNote(targetOp->getLoc())
        .append("target defined here");
  }

  return success();
}

// Parse the ContractOp syntax using the built-in parsing of function-like
// operations. We'll verify contract-specific restrictions in `verify`.
ParseResult ContractOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr typeAttrName = getFunctionTypeAttrName(result.name);
  StringAttr argAttrsName = getArgAttrsAttrName(result.name);

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
  if (parser.parseKeyword("for")) {
    return failure();
  }

  SymbolRefAttr targetAttr;
  if (parser.parseCustomAttributeWithFallback(
          targetAttr, parser.getBuilder().getType<::mlir::NoneType>()
      )) {
    return failure();
  }
  if (!targetAttr) {
    return failure();
  }
  result.getOrAddProperties<ContractOp::Properties>().target = targetAttr;

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
  Type type = builder.getFunctionType(argTypes, resultTypes);
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

  ContractOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}

void ContractOp::print(OpAsmPrinter &p) {
  // Print the operation and the contract name.
  p << ' ';
  p.printSymbolName(getSymName());

  // Print the name of the contract's target.
  p << " for ";
  p.printAttributeWithoutType(getTarget());
  p << ' ';

  ArrayRef<Type> argTypes = getArgumentTypes();
  function_interface_impl::printFunctionSignature(
      p, *this, argTypes, /*isVariadic*/ false, /*resultTypes*/ ArrayRef<Type>()
  );
  function_interface_impl::printFunctionAttributes(
      p, *this,
      /*elided*/ {getFunctionTypeAttrName(), getArgAttrsAttrName(), getTargetAttrName()}
  );
  // Print the body.
  Region &body = getRegion();
  p << ' ';
  p.printRegion(
      body, /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/false
  );
}

LogicalResult ContractOp::verify() {
  OwningEmitErrorFn emitErrorFunc = getEmitOpErrFn(this);

  if ((*this)->hasAttr(ARG_NAME_ATTR_NAME)) {
    return emitOpError() << '\'' << ARG_NAME_ATTR_NAME << "' is only valid on function arguments";
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
        return emitOpError() << '\'' << ARG_NAME_ATTR_NAME << "' on argument " << i
                             << " must be a string attribute";
      }
      if (!llvm::isa<NoneType>(argName.getType())) {
        return emitOpError() << '\'' << ARG_NAME_ATTR_NAME << "' on argument " << i
                             << " must not have an explicit type";
      }
      if (argName.getValue().empty()) {
        return emitOpError() << '\'' << ARG_NAME_ATTR_NAME << "' on argument " << i
                             << " must not be empty";
      }
      if (!seenNames.insert(argName).second) {
        return emitOpError() << "duplicate '" << ARG_NAME_ATTR_NAME << "' value \""
                             << argName.getValue() << "\" on argument " << i;
      }
    }
  }

  // Unlike for FuncDefOps, we don't verify that the inputs are valid LLZK types,
  // as we will check that the args match the target arguments in `verifySymbolUses()`

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

  return verifyRequireRestrictions(*this);
}

FailureOr<SymbolLookupResult<StructDefOp>>
ContractOp::getStructTarget(SymbolTableCollection &tables) {
  return lookupTopLevelSymbol<StructDefOp>(
      tables, getTarget(), getParentOfType<ModuleOp>(getOperation()), /*reportMissing*/ false
  );
}

FailureOr<SymbolLookupResult<FuncDefOp>> ContractOp::getFuncTarget(SymbolTableCollection &tables) {
  return lookupTopLevelSymbol<FuncDefOp>(
      tables, getTarget(), getParentOfType<ModuleOp>(getOperation()), /*reportMissing*/ false
  );
}

FailureOr<Value> ContractOp::getSelfValue() {
  if (failed(getStructTarget()) || getNumArguments() == 0) {
    return failure();
  }
  return getArgument(0);
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
    if (isDynamic(llvm::dyn_cast<IntegerAttr>(attr))) {
      continue;
    }
    auto it = unifications.find({FlatSymbolRefAttr::get(paramOp.getNameAttr()), Side::RHS});
    if (it != unifications.end() && !typeParamsUnify({attr}, {it->second})) {
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
    Operation *tgtOp = tgt.getOperation();
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
      // This is already checked by `verifyInputs()`, but `verifyTemplateParams()` is called
      // even if `verifyInputs()` fails for error aggregation, so we still need to return
      // early here.
      if (failed(unifyResult)) {
        return failure();
      }
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
      if (auto constParam = parent.getConstNamed<TemplateParamOp>(calleeAttr.getRootReference())) {
        return this->emitError("expected parameterized callee to target a struct function")
            .attachNote(constParam->getLoc())
            .append(
                " (i.e. \"@", FUNC_NAME_PRODUCT, "\", \"@", FUNC_NAME_COMPUTE, "\", or \"@",
                FUNC_NAME_CONSTRAIN, "\")"
            );
      }
    }
  }

  // Otherwise, callee must be specified via full path from the root module. Perform the full set of
  // checks against the known target function.
  auto tgtOpt = lookupTopLevelSymbol<ContractOp>(
      tables, calleeAttr, getParentOfType<ModuleOp>(getOperation())
  );
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
  }
  return failure();
}

FailureOr<SymbolLookupResult<ContractOp>>
IncludeOp::getCalleeTarget(SymbolTableCollection &tables) {
  Operation *thisOp = this->getOperation();
  auto root = getRootModule(thisOp);
  assert(succeeded(root));
  return llzk::lookupSymbolIn<ContractOp>(tables, getCallee(), root->getOperation(), thisOp);
}

bool IncludeOp::contractTargetsStruct() {
  SymbolTableCollection tables;
  auto callee = getCalleeTarget(tables);
  return succeeded(callee) && callee->get().hasStructTarget();
}

Value IncludeOp::getSelfValue() {
  SymbolTableCollection tables;
  auto callee = getCalleeTarget(tables);
  assert(succeeded(callee) && "include callee must resolve");
  if (!callee->get().hasStructTarget()) {
    return nullptr;
  }
  assert(getNumOperands() > 0 && "include op must have a self operand");
  return getOperand(0);
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
  output.insert(output.end(), input.begin(), input.end());
  return output;
}

Operation *IncludeOp::resolveCallableInTable(SymbolTableCollection *symbolTable) {
  FailureOr<SymbolLookupResult<ContractOp>> res =
      llzk::resolveCallable<ContractOp>(*symbolTable, *this);
  if (failed(res)) {
    return nullptr;
  }
  if (res->isManaged()) {
    this->emitWarning(
        "IncludeOp::resolveCallableInTable: cannot return "
        "pointer to a managed Operation since it would cause memory errors. "
        "Consider running -llzk-inline-includes to avoid encountering managed Operations."
    );
    return nullptr;
  }
  return res->get();
}

Operation *IncludeOp::resolveCallable() {
  SymbolTableCollection tables;
  return resolveCallableInTable(&tables);
}

} // namespace llzk::verif
