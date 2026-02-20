//===-- Ops.cpp - Func and call op implementations --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the LLVM Project's lib/Dialect/Func/IR/FuncOps.cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>

#include <llvm/ADT/MapVector.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Function/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::component;
using namespace llzk::polymorphic;

namespace llzk::function {

FunctionKind fnNameToKind(mlir::StringRef name) {
  if (FUNC_NAME_COMPUTE == name) {
    return FunctionKind::StructCompute;
  } else if (FUNC_NAME_CONSTRAIN == name) {
    return FunctionKind::StructConstrain;
  } else if (FUNC_NAME_PRODUCT == name) {
    return FunctionKind::StructProduct;
  } else {
    return FunctionKind::Free;
  }
}

namespace {
/// Ensure that all symbols used within the FunctionType can be resolved.
inline LogicalResult
verifyTypeResolution(SymbolTableCollection &tables, Operation *origin, FunctionType funcType) {
  return llzk::verifyTypeResolution(
      tables, origin, ArrayRef<ArrayRef<Type>> {funcType.getInputs(), funcType.getResults()}
  );
}
} // namespace

//===----------------------------------------------------------------------===//
// FuncDefOp
//===----------------------------------------------------------------------===//

FuncDefOp FuncDefOp::create(
    Location location, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs
) {
  return delegate_to_build<FuncDefOp>(location, name, type, attrs);
}

FuncDefOp FuncDefOp::create(
    Location location, StringRef name, FunctionType type, Operation::dialect_attr_range attrs
) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::ArrayRef(attrRef));
}

FuncDefOp FuncDefOp::create(
    Location location, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs,
    ArrayRef<DictionaryAttr> argAttrs
) {
  FuncDefOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void FuncDefOp::build(
    OpBuilder &builder, OperationState &state, StringRef name, FunctionType type,
    ArrayRef<NamedAttribute> attrs, ArrayRef<DictionaryAttr> argAttrs
) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name)
  );
}

ParseResult FuncDefOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType = [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
                          function_interface_impl::VariadicFlag,
                          std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name)
  );
}

void FuncDefOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName()
  );
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void FuncDefOp::cloneInto(FuncDefOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }
  for (const auto &attr : (*this)->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  auto newAttrs =
      llvm::to_vector(llvm::map_range(newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
    return NamedAttribute(attrPair.first, attrPair.second);
  }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
FuncDefOp FuncDefOp::clone(IRMapping &mapper) {
  // Create the new function.
  FuncDefOp newFunc = llvm::cast<FuncDefOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);
    for (unsigned i = 0; i != oldNumArgs; ++i) {
      if (!mapper.contains(getArgument(i))) {
        newInputs.push_back(oldType.getInput(i));
      }
    }

    /// If any of the arguments were dropped, update the type and drop any
    /// necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(FunctionType::get(oldType.getContext(), newInputs, oldType.getResults()));

      if (ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());
        for (unsigned i = 0; i != oldNumArgs; ++i) {
          if (!mapper.contains(getArgument(i))) {
            newArgAttrs.push_back(argAttrs[i]);
          }
        }
        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  /// Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}

FuncDefOp FuncDefOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

void FuncDefOp::setAllowConstraintAttr(bool newValue) {
  if (newValue) {
    getOperation()->setAttr(AllowConstraintAttr::name, UnitAttr::get(getContext()));
  } else {
    getOperation()->removeAttr(AllowConstraintAttr::name);
  }
}

void FuncDefOp::setAllowWitnessAttr(bool newValue) {
  if (newValue) {
    getOperation()->setAttr(AllowWitnessAttr::name, UnitAttr::get(getContext()));
  } else {
    getOperation()->removeAttr(AllowWitnessAttr::name);
  }
}

void FuncDefOp::setAllowNonNativeFieldOpsAttr(bool newValue) {
  if (newValue) {
    getOperation()->setAttr(AllowNonNativeFieldOpsAttr::name, UnitAttr::get(getContext()));
  } else {
    getOperation()->removeAttr(AllowNonNativeFieldOpsAttr::name);
  }
}

bool FuncDefOp::hasArgPublicAttr(unsigned index) {
  if (index < this->getNumArguments()) {
    DictionaryAttr res = function_interface_impl::getArgAttrDict(*this, index);
    return res ? res.contains(PublicAttr::name) : false;
  } else {
    // TODO: print error? requested attribute for non-existant argument index
    return false;
  }
}

LogicalResult FuncDefOp::verify() {
  OwningEmitErrorFn emitErrorFunc = getEmitOpErrFn(this);
  // Ensure that only valid LLZK types are used for arguments and return. Additionally, the struct
  // functions may not use AffineMapAttrs in their parameter types. If such a scenario seems to make
  // sense when generating LLZK IR, it's likely better to introduce a struct parameter to use
  // instead and instantiate the struct with that AffineMapAttr.
  FunctionType type = getFunctionType();
  for (Type t : type.getInputs()) {
    if (llzk::checkValidType(emitErrorFunc, t).failed()) {
      return failure();
    }
    if (isInStruct() && hasAffineMapAttr(t)) {
      return emitErrorFunc().append(
          "\"@", getName(), "\" parameters cannot contain affine map attributes but found ", t
      );
    }
  }
  for (Type t : type.getResults()) {
    if (llzk::checkValidType(emitErrorFunc, t).failed()) {
      return failure();
    }
  }
  // Ensure that the function does not contain nested modules.
  // Functions also cannot contain nested structs, but this check is handled
  // via struct.def's requirement of having module as a parent.
  WalkResult res = this->walk<WalkOrder::PreOrder>([this](ModuleOp nestedMod) {
    getEmitOpErrFn(nestedMod)().append(
        "cannot be nested within '", getOperation()->getName(), "' operations"
    );
    return WalkResult::interrupt();
  });
  if (res.wasInterrupted()) {
    return failure();
  }

  return success();
}

namespace {

LogicalResult
verifyFuncTypeCompute(FuncDefOp &origin, SymbolTableCollection &tables, StructDefOp &parent) {
  FunctionType funcType = origin.getFunctionType();
  llvm::ArrayRef<Type> resTypes = funcType.getResults();
  // Must return type of parent struct
  if (resTypes.size() != 1) {
    return origin.emitOpError().append(
        "\"@", FUNC_NAME_COMPUTE, "\" must have exactly one return type"
    );
  }
  if (failed(checkSelfType(tables, parent, resTypes.front(), origin, "return"))) {
    return failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. The return
  // types were already checked so just check the input types.
  return llzk::verifyTypeResolution(tables, origin, funcType.getInputs());
}

LogicalResult
verifyFuncTypeProduct(FuncDefOp &origin, SymbolTableCollection &tables, StructDefOp &parent) {
  // The signature for @product is the same as the signature for @compute
  return verifyFuncTypeCompute(origin, tables, parent);
}

LogicalResult
verifyFuncTypeConstrain(FuncDefOp &origin, SymbolTableCollection &tables, StructDefOp &parent) {
  FunctionType funcType = origin.getFunctionType();
  // Must return '()' type, i.e., have no return types
  if (funcType.getResults().size() != 0) {
    return origin.emitOpError() << "\"@" << FUNC_NAME_CONSTRAIN << "\" must have no return type";
  }

  // Type of the first parameter must match the parent StructDefOp of the current operation.
  llvm::ArrayRef<Type> inputTypes = funcType.getInputs();
  if (inputTypes.size() < 1) {
    return origin.emitOpError() << "\"@" << FUNC_NAME_CONSTRAIN
                                << "\" must have at least one input type";
  }
  if (failed(checkSelfType(tables, parent, inputTypes.front(), origin, "first input"))) {
    return failure();
  }

  // After the more specific checks (to ensure more specific error messages would be produced if
  // necessary), do the general check that all symbol references in the types are valid. There are
  // no return types, just check the remaining input types (the first was already checked via
  // the checkSelfType() call above).
  return llzk::verifyTypeResolution(tables, origin, inputTypes.drop_front());
}

} // namespace

LogicalResult FuncDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Additional checks for the compute/constrain/product functions within a struct
  FailureOr<StructDefOp> parentStructOpt = getParentOfType<StructDefOp>(*this);
  if (succeeded(parentStructOpt)) {
    // Verify return type restrictions for functions within a StructDefOp
    if (nameIsCompute()) {
      return verifyFuncTypeCompute(*this, tables, parentStructOpt.value());
    } else if (nameIsConstrain()) {
      return verifyFuncTypeConstrain(*this, tables, parentStructOpt.value());
    } else if (nameIsProduct()) {
      return verifyFuncTypeProduct(*this, tables, parentStructOpt.value());
    }
  }
  // In the general case, verify symbol resolution in all input and output types.
  return verifyTypeResolution(tables, *this, getFunctionType());
}

SymbolRefAttr FuncDefOp::getFullyQualifiedName(bool requireParent) {
  // If the parent is not present and not required, just return the symbol name
  if (!requireParent && getOperation()->getParentOp() == nullptr) {
    return SymbolRefAttr::get(getOperation());
  }
  auto res = getPathFromRoot(*this);
  assert(succeeded(res));
  return res.value();
}

Value FuncDefOp::getSelfValueFromCompute() {
  assert(nameIsCompute()); // skip inStruct check to allow dangling functions
  // Get the single block of the function body
  Region &body = getBody();
  assert(!body.empty() && "compute() function body is empty");
  Block &block = body.back();

  // The terminator should be the return op
  Operation *terminator = block.getTerminator();
  assert(terminator && "compute() function has no terminator");
  auto retOp = llvm::dyn_cast<ReturnOp>(terminator);
  if (!retOp) {
    llvm::errs() << "Expected '" << ReturnOp::getOperationName() << "' but found '"
                 << terminator->getName() << "'\n";
    llvm_unreachable("compute() function must end with ReturnOp");
  }
  return retOp.getOperands().front();
}

Value FuncDefOp::getSelfValueFromConstrain() {
  assert(nameIsConstrain()); // skip inStruct check to allow dangling functions
  return getArguments().front();
}

StructType FuncDefOp::getSingleResultTypeOfCompute() {
  assert(isStructCompute() && "violated implementation pre-condition");
  return getIfSingleton<StructType>(getResultTypes());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto function = getParentOp<FuncDefOp>(); // parent is FuncDefOp per ODS

  // The operand number and types must match the function signature.
  const auto results = function.getFunctionType().getResults();
  if (getNumOperands() != results.size()) {
    return emitOpError("has ") << getNumOperands() << " operands, but enclosing function (@"
                               << function.getName() << ") returns " << results.size();
  }

  for (unsigned i = 0, e = results.size(); i != e; ++i) {
    if (!typesUnify(getOperand(i).getType(), results[i])) {
      return emitError() << "type of return operand " << i << " (" << getOperand(i).getType()
                         << ") doesn't match function result type (" << results[i] << ")"
                         << " in function @" << function.getName();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

void CallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, SymbolRefAttr callee,
    ValueRange argOperands
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrsEmpty<CallOp>(
      odsBuilder, odsState, static_cast<int32_t>(argOperands.size())
  );
  props.setCallee(callee);
}

void CallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, SymbolRefAttr callee,
    ArrayRef<ValueRange> mapOperands, DenseI32ArrayAttr numDimsPerMap, ValueRange argOperands
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrs<CallOp>(
      odsBuilder, odsState, mapOperands, numDimsPerMap, argOperands.size()
  );
  props.setCallee(callee);
}

namespace {

struct CallOpVerifier {
  CallOpVerifier(CallOp *c, StringRef tgtName) : callOp(c), tgtKind(fnNameToKind(tgtName)) {}
  virtual ~CallOpVerifier() = default;

  LogicalResult verify() {
    // Rather than immediately returning on failure, we check all verifier steps and aggregate to
    // provide as many errors are possible in a single verifier run.
    LogicalResult aggregateResult = success();
    if (failed(verifyTargetAttributes())) {
      aggregateResult = failure();
    }
    if (failed(verifyInputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyOutputs())) {
      aggregateResult = failure();
    }
    if (failed(verifyAffineMapParams())) {
      aggregateResult = failure();
    }
    return aggregateResult;
  }

protected:
  CallOp *callOp;
  FunctionKind tgtKind;

  virtual LogicalResult verifyTargetAttributes() = 0;
  virtual LogicalResult verifyInputs() = 0;
  virtual LogicalResult verifyOutputs() = 0;
  virtual LogicalResult verifyAffineMapParams() = 0;

  /// Ensure that if the target allows witness/constraint ops, the caller does as well.
  LogicalResult verifyTargetAttributesMatch(FuncDefOp target) {
    LogicalResult aggregateRes = success();
    if (FuncDefOp caller = (*callOp)->getParentOfType<FuncDefOp>()) {
      auto emitAttrErr = [&](StringLiteral attrName) {
        aggregateRes = callOp->emitOpError()
                       << "target '@" << target.getName() << "' has '" << attrName
                       << "' attribute, which is not specified by the caller '@" << caller.getName()
                       << '\'';
      };

      if (target.hasAllowConstraintAttr() && !caller.hasAllowConstraintAttr()) {
        emitAttrErr(AllowConstraintAttr::name);
      }
      if (target.hasAllowWitnessAttr() && !caller.hasAllowWitnessAttr()) {
        emitAttrErr(AllowWitnessAttr::name);
      }
      if (target.hasAllowNonNativeFieldOpsAttr() && !caller.hasAllowNonNativeFieldOpsAttr()) {
        emitAttrErr(AllowNonNativeFieldOpsAttr::name);
      }
    }
    return aggregateRes;
  }

  LogicalResult verifyNoAffineMapInstantiations() {
    if (!isNullOrEmpty(callOp->getMapOpGroupSizesAttr())) {
      // Tested in call_with_affinemap_fail.llzk
      return callOp->emitOpError().append(
          "can only have affine map instantiations when targeting a \"@", FUNC_NAME_COMPUTE,
          "\" function"
      );
    }
    // ASSERT: the check above is sufficient due to VerifySizesForMultiAffineOps trait.
    assert(isNullOrEmpty(callOp->getNumDimsPerMapAttr()));
    assert(callOp->getMapOperands().empty());
    return success();
  }
};

struct KnownTargetVerifier : public CallOpVerifier {
  KnownTargetVerifier(CallOp *c, SymbolLookupResult<FuncDefOp> &&tgtRes)
      : CallOpVerifier(c, tgtRes.get().getSymName()), tgt(*tgtRes), tgtType(tgt.getFunctionType()),
        includeSymNames(tgtRes.getNamespace()) {}

  LogicalResult verifyTargetAttributes() override {
    return CallOpVerifier::verifyTargetAttributesMatch(tgt);
  }

  LogicalResult verifyInputs() override {
    return verifyTypesMatch(callOp->getArgOperands().getTypes(), tgtType.getInputs(), "operand");
  }

  LogicalResult verifyOutputs() override {
    return verifyTypesMatch(callOp->getResultTypes(), tgtType.getResults(), "result");
  }

  LogicalResult verifyAffineMapParams() override {
    if ((FunctionKind::StructCompute == tgtKind || FunctionKind::StructProduct == tgtKind) &&
        isInStruct(tgt.getOperation())) {
      // Return type should be a single StructType. If that is not the case here, just bail without
      // producing an error. The combination of this KnownTargetVerifier resolving the callee to a
      // specific FuncDefOp and verifyFuncTypeCompute() ensuring all FUNC_NAME_COMPUTE FuncOps have
      // a single StructType return value will produce a more relevant error message in that case.
      if (StructType retTy = callOp->getSingleResultTypeOfWitnessGen()) {
        if (ArrayAttr params = retTy.getParams()) {
          // Collect the struct parameters that are defined via AffineMapAttr
          SmallVector<AffineMapAttr> mapAttrs;
          for (Attribute a : params) {
            if (AffineMapAttr m = dyn_cast<AffineMapAttr>(a)) {
              mapAttrs.push_back(m);
            }
          }
          return affineMapHelpers::verifyAffineMapInstantiations(
              callOp->getMapOperands(), callOp->getNumDimsPerMap(), mapAttrs, *callOp
          );
        }
      }
      return success();
    } else {
      // Global functions and constrain functions cannot have affine map instantiations.
      return verifyNoAffineMapInstantiations();
    }
  }

private:
  template <typename T>
  LogicalResult
  verifyTypesMatch(ValueTypeRange<T> callOpTypes, ArrayRef<Type> tgtTypes, const char *aspect) {
    if (tgtTypes.size() != callOpTypes.size()) {
      return callOp->emitOpError()
          .append("incorrect number of ", aspect, "s for callee, expected ", tgtTypes.size())
          .attachNote(tgt.getLoc())
          .append("callee defined here");
    }
    for (unsigned i = 0, e = tgtTypes.size(); i != e; ++i) {
      if (!typesUnify(callOpTypes[i], tgtTypes[i], includeSymNames)) {
        return callOp->emitOpError().append(
            aspect, " type mismatch: expected type ", tgtTypes[i], ", but found ", callOpTypes[i],
            " for ", aspect, " number ", i
        );
      }
    }
    return success();
  }

  FuncDefOp tgt;
  FunctionType tgtType;
  std::vector<llvm::StringRef> includeSymNames;
};

/// Version of checkSelfType() that performs the subset of verification checks that can be done when
/// the exact target of the `CallOp` is unknown.
LogicalResult checkSelfTypeUnknownTarget(
    StringAttr expectedParamName, Type actualType, CallOp *origin, const char *aspect
) {
  if (!llvm::isa<TypeVarType>(actualType) ||
      llvm::cast<TypeVarType>(actualType).getRefName() != expectedParamName) {
    // Tested in function_restrictions_fail.llzk:
    //    Non-tvar for constrain input via "call_target_constrain_without_self_non_struct"
    //    Non-tvar for compute output via "call_target_compute_wrong_type_ret"
    //    Wrong tvar for constrain input via "call_target_constrain_without_self_wrong_tvar_param"
    //    Wrong tvar for compute output via "call_target_compute_wrong_tvar_param_ret"
    return origin->emitOpError().append(
        "target \"@", origin->getCallee().getLeafReference().getValue(), "\" expected ", aspect,
        " type '!", TypeVarType::name, "<@", expectedParamName.getValue(), ">' but found ",
        actualType
    );
  }
  return success();
}

/// Precondition: the CallOp callee references a parameter of the CallOp's parent struct. This
/// creates a restriction that the referenced parameter must be instantiated with a StructType.
/// Hence, the call must target a function within a struct, not a global function, so the callee
/// name must be `compute`, `constrain`, or `product`, nothing else.
/// Normally, full verification of the `compute` and `constrain` callees is done via
/// KnownTargetVerifier, which checks that input and output types of the caller match the callee,
/// plus verifyFuncTypeCompute() when the callee is `compute` or verifyFuncTypeConstrain() when
/// the callee is `constrain`. Those checks can take place after all parameterized structs are
/// instantiated (and thus the call target is known). For now, only minimal checks can be done.
struct UnknownTargetVerifier : public CallOpVerifier {
  UnknownTargetVerifier(CallOp *c, SymbolRefAttr callee)
      : CallOpVerifier(c, callee.getLeafReference().getValue()), calleeAttr(callee) {}

  LogicalResult verifyTargetAttributes() override {
    // Based on the precondition of this verifier, the target must be either a
    // struct compute, constrain, or product function.
    LogicalResult aggregateRes = success();
    if (FuncDefOp caller = (*callOp)->getParentOfType<FuncDefOp>()) {
      auto emitAttrErr = [&](StringLiteral attrName) {
        aggregateRes = callOp->emitOpError()
                       << "target '" << calleeAttr << "' has '" << attrName
                       << "' attribute, which is not specified by the caller '@" << caller.getName()
                       << '\'';
      };

      switch (tgtKind) {
      case FunctionKind::StructConstrain:
        if (!caller.hasAllowConstraintAttr()) {
          emitAttrErr(AllowConstraintAttr::name);
        }
        break;
      case FunctionKind::StructCompute:
        if (!caller.hasAllowWitnessAttr()) {
          emitAttrErr(AllowWitnessAttr::name);
        }
        break;
      case FunctionKind::StructProduct:
        if (!caller.hasAllowWitnessAttr()) {
          emitAttrErr(AllowWitnessAttr::name);
        }
        if (!caller.hasAllowConstraintAttr()) {
          emitAttrErr(AllowConstraintAttr::name);
        }
        break;
      default:
        break;
      }
    }
    return aggregateRes;
  }

  LogicalResult verifyInputs() override {
    if (FunctionKind::StructCompute == tgtKind || FunctionKind::StructProduct == tgtKind) {
      // Without known target, no additional checks can be done.
    } else if (FunctionKind::StructConstrain == tgtKind) {
      // Without known target, this can only check that the first input is VarType using the same
      // struct parameter as the base of the callee (later replaced with the target struct's type).
      Operation::operand_type_range inputTypes = callOp->getArgOperands().getTypes();
      if (inputTypes.size() < 1) {
        // Tested in function_restrictions_fail.llzk
        return callOp->emitOpError()
               << "target \"@" << FUNC_NAME_CONSTRAIN << "\" must have at least one input type";
      }
      return checkSelfTypeUnknownTarget(
          calleeAttr.getRootReference(), inputTypes.front(), callOp, "first input"
      );
    }
    return success();
  }

  LogicalResult verifyOutputs() override {
    if (FunctionKind::StructCompute == tgtKind || FunctionKind::StructProduct == tgtKind) {
      // Without known target, this can only check that the function returns VarType using the same
      // struct parameter as the base of the callee (later replaced with the target struct's type).
      Operation::result_type_range resTypes = callOp->getResultTypes();
      if (resTypes.size() != 1) {
        // Tested in function_restrictions_fail.llzk
        return callOp->emitOpError().append(
            "target \"@", FUNC_NAME_COMPUTE, "\" must have exactly one return type"
        );
      }
      return checkSelfTypeUnknownTarget(
          calleeAttr.getRootReference(), resTypes.front(), callOp, "return"
      );
    } else if (FunctionKind::StructConstrain == tgtKind) {
      // Without known target, this can only check that the function has no return
      if (callOp->getNumResults() != 0) {
        // Tested in function_restrictions_fail.llzk
        return callOp->emitOpError()
               << "target \"@" << FUNC_NAME_CONSTRAIN << "\" must have no return type";
      }
    }
    return success();
  }

  LogicalResult verifyAffineMapParams() override {
    if (FunctionKind::StructCompute == tgtKind || FunctionKind::StructProduct == tgtKind) {
      // Without known target, no additional checks can be done.
    } else if (FunctionKind::StructConstrain == tgtKind) {
      // Without known target, this can only check that there are no affine map instantiations.
      return verifyNoAffineMapInstantiations();
    }
    return success();
  }

private:
  SymbolRefAttr calleeAttr;
};

} // namespace

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &tables) {
  // First, verify symbol resolution in all input and output types.
  if (failed(verifyTypeResolution(tables, *this, getCalleeType()))) {
    return failure(); // verifyTypeResolution() already emits a sufficient error message
  }

  // Check that the callee attribute was specified.
  SymbolRefAttr calleeAttr = getCalleeAttr();
  if (!calleeAttr) {
    return emitOpError("requires a 'callee' symbol reference attribute");
  }

  // If the callee references a parameter of the struct where this call appears, perform the subset
  // of checks that can be done even though the target is unknown.
  if (calleeAttr.getNestedReferences().size() == 1) {
    FailureOr<StructDefOp> parent = getParentOfType<StructDefOp>(*this);
    if (succeeded(parent) && parent->hasParamNamed(calleeAttr.getRootReference())) {
      return UnknownTargetVerifier(this, calleeAttr).verify();
    }
  }

  // Otherwise, callee must be specified via full path from the root module. Perform the full set of
  // checks against the known target function.
  auto tgtOpt = lookupTopLevelSymbol<FuncDefOp>(tables, calleeAttr, *this);
  if (failed(tgtOpt)) {
    return this->emitError() << "expected '" << FuncDefOp::getOperationName() << "' named \""
                             << calleeAttr << '"';
  }
  return KnownTargetVerifier(this, std::move(*tgtOpt)).verify();
}

FunctionType CallOp::getCalleeType() {
  return FunctionType::get(getContext(), getArgOperands().getTypes(), getResultTypes());
}

namespace {

bool calleeIsStructFunctionImpl(
    const char *funcName, SymbolRefAttr callee, llvm::function_ref<StructType()> getType
) {
  if (callee.getLeafReference() == funcName) {
    if (StructType t = getType()) {
      // If the name ref within the StructType matches the `callee` prefix (i.e., sans the function
      // name itself), then the `callee` target must be within a StructDefOp because validation
      // checks elsewhere ensure that every StructType references a StructDefOp (i.e., the `callee`
      // function is not simply a global function nested within a ModuleOp)
      return t.getNameRef() == getPrefixAsSymbolRefAttr(callee);
    }
  }
  return false;
}

} // namespace

bool CallOp::calleeIsStructCompute() {
  return calleeIsStructFunctionImpl(FUNC_NAME_COMPUTE, getCallee(), [this]() {
    return this->getSingleResultTypeOfCompute();
  });
}

bool CallOp::calleeIsStructConstrain() {
  return calleeIsStructFunctionImpl(FUNC_NAME_CONSTRAIN, getCallee(), [this]() {
    return getAtIndex<StructType>(this->getArgOperands().getTypes(), 0);
  });
}

Value CallOp::getSelfValueFromCompute() {
  assert(calleeIsStructCompute());
  return getResults().front();
}

Value CallOp::getSelfValueFromConstrain() {
  assert(calleeIsStructConstrain());
  return getArgOperands().front();
}

FailureOr<SymbolLookupResult<FuncDefOp>> CallOp::getCalleeTarget(SymbolTableCollection &tables) {
  Operation *thisOp = this->getOperation();
  auto root = getRootModule(thisOp);
  assert(succeeded(root));
  return llzk::lookupSymbolIn<FuncDefOp>(tables, getCallee(), root->getOperation(), thisOp);
}

StructType CallOp::getSingleResultTypeOfCompute() {
  assert(calleeIsCompute() && "violated implementation pre-condition");
  return getIfSingleton<StructType>(getResultTypes());
}

StructType CallOp::getSingleResultTypeOfWitnessGen() {
  assert(calleeContainsWitnessGen() && "violated implementation pre-condition");
  return getIfSingleton<StructType>(getResultTypes());
}

/// Return the callee of this operation.
CallInterfaceCallable CallOp::getCallableForCallee() { return getCalleeAttr(); }

/// Set the callee for this operation.
void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  setCalleeAttr(llvm::cast<SymbolRefAttr>(callee));
}

SmallVector<ValueRange> CallOp::toVectorOfValueRange(OperandRangeRange input) {
  llvm::SmallVector<ValueRange, 4> output;
  output.reserve(input.size());
  for (OperandRange r : input) {
    output.push_back(r);
  }
  return output;
}

Operation *CallOp::resolveCallableInTable(SymbolTableCollection *symbolTable) {
  FailureOr<SymbolLookupResult<FuncDefOp>> res =
      llzk::resolveCallable<FuncDefOp>(*symbolTable, *this);
  if (LogicalResult(res).failed() || res->isManaged()) {
    // Cannot return pointer to a managed Operation since it would cause memory errors.
    return nullptr;
  }
  return res->get();
}

Operation *CallOp::resolveCallable() {
  SymbolTableCollection tables;
  return resolveCallableInTable(&tables);
}

} // namespace llzk::function
