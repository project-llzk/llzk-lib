//===-- Ops.cpp - Func and call op implementations --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the LLVM Project's lib/Dialect/Func/IR/FuncOps.cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Function/IR/Ops.h"

#include "llzk/Dialect/Felt/IR/Attrs.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Dialect.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/Interfaces/FunctionImplementation.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/MapVector.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Function/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::felt;
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

/// Verify the name attributes on function arguments or results.
/// ownAttrName/ownLabel describe the attribute valid on this side (e.g. RES_NAME_ATTR_NAME /
/// "result"), while crossAttrName/crossLabel describe the attribute that belongs on the other
/// side and must not appear here.
static LogicalResult verifyArgOrResNameAttrs(
    ArrayAttr attrs, StringRef ownAttrName, StringRef crossAttrName, StringRef ownLabel,
    StringRef crossLabel, EmitErrorFn emitFn
) {
  if (!attrs) {
    return success();
  }
  llvm::DenseSet<StringAttr> seenNames;
  for (auto [i, attr] : llvm::enumerate(attrs)) {
    auto dictAttr = llvm::dyn_cast<DictionaryAttr>(attr);
    if (!dictAttr) {
      continue;
    }
    if (dictAttr.contains(crossAttrName)) {
      return emitFn().append(
          '\'', crossAttrName, "' is only valid on function ", crossLabel, "s but found on ",
          ownLabel, ' ', i
      );
    }
    Attribute nameAttr = dictAttr.get(ownAttrName);
    if (!nameAttr) {
      continue;
    }
    auto name = llvm::dyn_cast<StringAttr>(nameAttr);
    if (!name) {
      return emitFn().append(
          '\'', ownAttrName, "' on ", ownLabel, ' ', i, " must be a string attribute"
      );
    }
    if (!llvm::isa<NoneType>(name.getType())) {
      return emitFn().append(
          '\'', ownAttrName, "' on ", ownLabel, ' ', i, " must not have an explicit type"
      );
    }
    if (name.getValue().empty()) {
      return emitFn().append('\'', ownAttrName, "' on ", ownLabel, ' ', i, " must not be empty");
    }
    if (!seenNames.insert(name).second) {
      return emitFn().append(
          "duplicate '", ownAttrName, "' value \"", name.getValue(), "\" on ", ownLabel, ' ', i
      );
    }
  }
  return success();
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

bool FuncDefOp::hasArgName(unsigned index) { return static_cast<bool>(getArgNameAttr(index)); }

std::optional<StringAttr> FuncDefOp::getArgNameAttr(unsigned index) {
  if (index >= getNumArguments()) {
    return std::nullopt;
  }
  if (StringAttr attr = getArgAttrOfType<StringAttr>(index, ARG_NAME_ATTR_NAME)) {
    return attr;
  }
  return std::nullopt;
}

void FuncDefOp::setArgNameAttr(unsigned index, const StringAttr &attr) {
  assert(index < getNumArguments() && "argument index out of range");
  setArgAttr(index, ARG_NAME_ATTR_NAME, attr);
}

void FuncDefOp::setArgName(unsigned index, StringRef name) {
  setArgNameAttr(index, StringAttr::get(getContext(), name));
}

LogicalResult FuncDefOp::verify() {
  OwningEmitErrorFn emitErrorFunc = getEmitOpErrFn(this);

  if ((*this)->hasAttr(ARG_NAME_ATTR_NAME)) {
    return emitErrorFunc() << '\'' << ARG_NAME_ATTR_NAME << "' is only valid on function arguments";
  }
  if ((*this)->hasAttr(RES_NAME_ATTR_NAME)) {
    return emitErrorFunc() << '\'' << RES_NAME_ATTR_NAME << "' is only valid on function results";
  }

  if (failed(verifyArgOrResNameAttrs(
          getAllResultAttrs(), RES_NAME_ATTR_NAME, ARG_NAME_ATTR_NAME, "result", "argument",
          emitErrorFunc
      ))) {
    return failure();
  }

  if (failed(verifyArgOrResNameAttrs(
          getAllArgAttrs(), ARG_NAME_ATTR_NAME, RES_NAME_ATTR_NAME, "argument", "result",
          emitErrorFunc
      ))) {
    return failure();
  }

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
  if (StructDefOp parentStructOpt = getParentOfType<StructDefOp>(*this)) {
    // Verify return type restrictions for functions within a StructDefOp
    if (nameIsCompute()) {
      return verifyFuncTypeCompute(*this, tables, parentStructOpt);
    } else if (nameIsConstrain()) {
      return verifyFuncTypeConstrain(*this, tables, parentStructOpt);
    } else if (nameIsProduct()) {
      return verifyFuncTypeProduct(*this, tables, parentStructOpt);
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

// Custom implementation to deserialize bytecode produced prior to version 2 which added optional
// `OptionalAttr<ArrayAttr>:$templateParams`.
LogicalResult CallOp::readProperties(DialectBytecodeReader &reader, OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>();
  if (failed(reader.readAttribute(prop.callee)) ||
      failed(reader.readAttribute(prop.mapOpGroupSizes)) ||
      failed(reader.readOptionalAttribute(prop.numDimsPerMap))) {
    return failure();
  }

  if (reader.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    DenseI32ArrayAttr attr;
    if (failed(reader.readAttribute(attr))) {
      return failure();
    }
    if (attr.size() > static_cast<int64_t>(sizeof(propStorage) / sizeof(int32_t))) {
      reader.emitError("size mismatch for operand/result_segment_size");
      return failure();
    }
    llvm::copy(ArrayRef<int32_t>(attr), propStorage.begin());
  }

  // The `templateParams` is only available in version 2 or later.
  auto versionOpt = reader.getDialectVersion<FunctionDialect>();
  if (succeeded(versionOpt)) {
    const auto &ver = static_cast<const LLZKDialectVersion &>(**versionOpt);
    if (ver.majorVersion >= 2) {
      if (failed(reader.readOptionalAttribute(prop.templateParams))) {
        return failure();
      }
    }
  }

  if (reader.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6) {
    return reader.readSparseArray(MutableArrayRef(prop.operandSegmentSizes));
  };
  return success();
}

// Same as tablegen would generate to serialize current version IR.
void CallOp::writeProperties(DialectBytecodeWriter &writer) {
  auto &prop = getProperties();
  writer.writeAttribute(prop.callee);
  writer.writeAttribute(prop.mapOpGroupSizes);
  writer.writeOptionalAttribute(prop.numDimsPerMap);

  if (writer.getBytecodeVersion() < /*kNativePropertiesODSSegmentSize=*/6) {
    auto &propStorage = prop.operandSegmentSizes;
    writer.writeAttribute(DenseI32ArrayAttr::get(this->getContext(), propStorage));
  }

  writer.writeOptionalAttribute(prop.templateParams);

  auto &propStorage = prop.operandSegmentSizes;
  if (writer.getBytecodeVersion() >= /*kNativePropertiesODSSegmentSize=*/6) {
    writer.writeSparseArray(ArrayRef(propStorage));
  }
}

void CallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, SymbolRefAttr callee,
    ValueRange argOperands, ArrayRef<Attribute> templateParams
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrsEmpty<CallOp>(
      odsBuilder, odsState, llzk::checkedCast<int32_t>(argOperands.size())
  );
  props.setCallee(callee);
  addTemplateParams<CallOp>(odsBuilder, props, templateParams);
}

void CallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, SymbolRefAttr callee,
    ArrayRef<ValueRange> mapOperands, DenseI32ArrayAttr numDimsPerMap, ValueRange argOperands,
    ArrayRef<Attribute> templateParams
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(argOperands);
  Properties &props = affineMapHelpers::buildInstantiationAttrs<CallOp>(
      odsBuilder, odsState, mapOperands, numDimsPerMap,
      llzk::checkedCast<int32_t>(argOperands.size())
  );
  props.setCallee(callee);
  addTemplateParams<CallOp>(odsBuilder, props, templateParams);
}

LogicalResult
CallOp::verifyTemplateParamCompatibility(Attribute paramFromCallOp, TemplateParamOp targetParam) {
  // A wildcard `?` (represented as kDynamic) defers inference to a later pass.
  // It is only valid for parameters with a `!poly.tvar` type restriction.
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(paramFromCallOp)) {
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
    bool compatible = false;
    if (auto sym = llvm::dyn_cast<SymbolRefAttr>(paramFromCallOp)) {
      if (sym.getNestedReferences().empty()) {
        SymbolTableCollection tables;
        FailureOr<TemplateOp> parentTemplate = getConstResolutionTemplate(tables, *this);
        if (failed(parentTemplate)) {
          return failure();
        }
        if (TemplateOp p = *parentTemplate) {
          auto binding = p.getConstNamed<TemplateSymbolBindingOpInterface>(sym.getRootReference());
          if (binding) {
            // Once we know it references a template symbol binding, assume it's compatible unless
            // the optional type is present and doesn't unify with the declared type.
            if (std::optional<Type> actualType = binding.getTypeOpt()) {
              compatible = typesUnify(*actualType, *declaredType);
            } else {
              compatible = true;
            }
          }
        }
      }
    } else if (llvm::isa<TypeVarType>(*declaredType)) {
      compatible = llvm::isa<TypeAttr>(paramFromCallOp);
    } else if (llvm::isa<FeltType>(*declaredType)) {
      compatible = llvm::isa<FeltConstAttr, IntegerAttr>(paramFromCallOp) &&
                   isValidConstReadType(llvm::cast<TypedAttr>(paramFromCallOp).getType());
    } else if (llvm::isa<IndexType, IntegerType>(*declaredType)) {
      // Note: Just like struct type instantiation, there is no restriction on passing a
      // larger value to an `i1`. The flattening pass will treat 0 as false and any other
      // value as true (but give a warning if it's not 1).
      compatible = llvm::isa<IntegerAttr>(paramFromCallOp) &&
                   isValidConstReadType(llvm::cast<TypedAttr>(paramFromCallOp).getType());
    } else {
      // Note: `declaredType` is restricted by `isValidConstReadType()`
      llvm_unreachable("inconsistent with `isValidConstReadType()`");
    }
    if (!compatible) {
      // Tested in call_with_template_params_fail.llzk
      return this->emitOpError().append(
          "instantiation value '", paramFromCallOp, "' is not compatible with parameter \"@",
          targetParam.getName(), "\" type restriction ", *declaredType
      );
    }
  }
  return success();
}

LogicalResult CallOp::verifyTemplateParamCompatibility(
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

LogicalResult CallOp::verifyTemplateParamsMatchInferred(
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

struct CallOpVerifier {
  CallOpVerifier(CallOp *c, FunctionKind tgtFuncKind) : callOp(c), tgtKind(tgtFuncKind) {}
  CallOpVerifier(CallOp *c, StringRef tgtName) : CallOpVerifier(c, fnNameToKind(tgtName)) {}
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
    if (failed(verifyTemplateParams())) {
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
  virtual LogicalResult verifyTemplateParams() = 0;
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

  LogicalResult verifyNoTemplateInstantiations() {
    if (!isNullOrEmpty(callOp->getTemplateParamsAttr())) {
      // Tested in call_with_template_params_fail.llzk
      return callOp->emitOpError().append(
          "can only have template instantiations when targeting a templated free function"
      );
    }
    return success();
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

  LogicalResult verifyTemplateParams() override {
    Operation *tgtOp = tgt.getOperation();
    if (isInStruct(tgtOp)) {
      // Struct function calls cannot contain template parameter instantiations.
      return verifyNoTemplateInstantiations();
    } else if (TemplateOp tgtOpParent = getParentOfType<TemplateOp>(tgtOp)) {
      // When the target function is a free function within a TemplateOp, the CallOp may have
      // template parameter instantiations that must be checked against the template parameters.
      // - If the function type signature references all template parameters, then the parameter
      //   instantiation list on the CallOp is optional, otherwise it's required.
      // - If present, the instantiation list must provide a value for every template parameter
      //   and the value must be type-compatible with the parameter's declared type (if any).
      // - If present, the instantiation list must result in a function type signature that can
      //   be unified with the CallOp's operand and result types.
      auto realParams = tgtOpParent.getConstOps<TemplateParamOp>();
      ArrayAttr callParams = callOp->getTemplateParamsAttr();

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
        return callOp->emitOpError().append(
            "must provide template instantiation parameters when calling \"@", tgt.getSymName(),
            "\" because not all template parameters of \"@", tgtOpParent.getSymName(),
            "\" appear in the function type signature"
        );
      }

      // Ensure `forceIntAttrTypes()` was successful on the CallOp's template parameters.
      if (failed(llzk::forceIntAttrTypes(callParams.getValue(), [this] {
        return llzk::InFlightDiagnosticWrapper(this->callOp->emitOpError());
      }))) {
        return failure();
      }

      // The instantiation list is present. Check it has exactly one entry per template param.
      size_t numTemplateParams = llvm::range_size(realParams);
      if (callParams.size() != numTemplateParams) {
        // Tested in call_with_template_params_fail.llzk
        return callOp->emitOpError().append(
            "template instantiation has ", callParams.size(), " parameter(s) but \"@",
            tgtOpParent.getSymName(), "\" expects ", numTemplateParams, " template parameter(s)"
        );
      }

      // Check type compatibility of each provided value with the declared parameter type (if any).
      if (failed(callOp->verifyTemplateParamCompatibility(realParams))) {
        return failure();
      }

      // Check that the provided instantiation values are consistent with what type unification
      // of the target function types against the call's operand and result types would determine.
      FailureOr<UnificationMap> unifyResult = callOp->unifyTypeSignature(tgtType);
      assert(succeeded(unifyResult) && "already checked by `verifyInputs()` and `verifyOutputs()`");
      return callOp->verifyTemplateParamsMatchInferred(realParams, unifyResult.value());
    } else {
      // Non-template functions cannot contain template parameter instantiations.
      return verifyNoTemplateInstantiations();
    }
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

/// Precondition: The CallOp callee root symbol ref is a parameter of the CallOp's parent template.
/// This creates a restriction that the referenced template parameter must be instantiated with a
/// StructType. Hence, the call must target a function within a struct (i.e. not a free function),
/// so the callee name must be `compute`, `constrain`, or `product`, nothing else. Normally, full
/// verification of the `compute` and `constrain` callees is done via KnownTargetVerifier, which
/// checks that input and output types of the caller match the callee, plus verifyFuncTypeCompute()
/// when the callee is `compute` or verifyFuncTypeConstrain() when the callee is `constrain`. Those
/// checks can take place after all parameterized structs are instantiated (and thus the call target
/// is known). For now, only minimal checks can be done.
struct UnknownTargetVerifier : public CallOpVerifier {
  UnknownTargetVerifier(CallOp *c, FunctionKind tgtFuncKind, SymbolRefAttr callee)
      : CallOpVerifier(c, tgtFuncKind), calleeAttr(callee) {
    assert(
        tgtFuncKind == FunctionKind::StructCompute ||
        tgtFuncKind == FunctionKind::StructConstrain || tgtFuncKind == FunctionKind::StructProduct
    ); // pre-condition mentioned above
  }

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

  LogicalResult verifyTemplateParams() override {
    // Struct function calls cannot contain template parameter instantiations.
    return verifyNoTemplateInstantiations();
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
        FunctionKind tgtKind = fnNameToKind(calleeAttr.getLeafReference().getValue());
        if (tgtKind != FunctionKind::Free) {
          return UnknownTargetVerifier(this, tgtKind, calleeAttr).verify();
        }
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
  auto tgtOpt = lookupTopLevelSymbol<FuncDefOp>(tables, calleeAttr, *this);
  if (failed(tgtOpt)) {
    return this->emitError() << "expected '" << FuncDefOp::getOperationName() << "' named \""
                             << calleeAttr << '"';
  }
  return KnownTargetVerifier(this, std::move(*tgtOpt)).verify();
}

FunctionType CallOp::getTypeSignature() {
  return FunctionType::get(getContext(), getArgOperands().getTypes(), getResultTypes());
}

FailureOr<UnificationMap> CallOp::unifyTypeSignature(FunctionType other) {
  UnificationMap unifications;
  if (functionTypesUnify(getTypeSignature(), other, {}, &unifications)) {
    return unifications;
  } else {
    return failure();
  }
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
      // function is not simply a free function nested within a ModuleOp)
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

bool CallOp::calleeIsStructProduct() {
  return calleeIsStructFunctionImpl(FUNC_NAME_PRODUCT, getCallee(), [this]() {
    return this->getSingleResultTypeOfWitnessGen();
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
  if (failed(res) || res->isManaged()) {
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
