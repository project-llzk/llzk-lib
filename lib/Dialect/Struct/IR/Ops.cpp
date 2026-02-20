//===-- Ops.cpp - Struct op implementations ---------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/IR/IRMapping.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSet.h>

#include <optional>

// TableGen'd implementation files
#include "llzk/Dialect/Struct/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Struct/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::felt;
using namespace llzk::array;
using namespace llzk::felt;
using namespace llzk::function;

namespace llzk::component {

bool isInStruct(Operation *op) { return succeeded(getParentOfType<StructDefOp>(op)); }

FailureOr<StructDefOp> verifyInStruct(Operation *op) {
  FailureOr<StructDefOp> res = getParentOfType<StructDefOp>(op);
  if (failed(res)) {
    return op->emitOpError() << "only valid within a '" << StructDefOp::getOperationName()
                             << "' ancestor";
  }
  return res;
}

bool isInStructFunctionNamed(Operation *op, char const *funcName) {
  FailureOr<FuncDefOp> parentFuncOpt = getParentOfType<FuncDefOp>(op);
  if (succeeded(parentFuncOpt)) {
    FuncDefOp parentFunc = parentFuncOpt.value();
    if (isInStruct(parentFunc.getOperation())) {
      if (parentFunc.getSymName().compare(funcName) == 0) {
        return true;
      }
    }
  }
  return false;
}

// Again, only valid/implemented for StructDefOp
template <> LogicalResult SetFuncAllowAttrs<StructDefOp>::verifyTrait(Operation *structOp) {
  assert(llvm::isa<StructDefOp>(structOp));
  Region &bodyRegion = llvm::cast<StructDefOp>(structOp).getBodyRegion();
  if (!bodyRegion.empty()) {
    bodyRegion.front().walk([](FuncDefOp funcDef) {
      if (funcDef.nameIsConstrain()) {
        funcDef.setAllowConstraintAttr();
        funcDef.setAllowWitnessAttr(false);
      } else if (funcDef.nameIsCompute()) {
        funcDef.setAllowConstraintAttr(false);
        funcDef.setAllowWitnessAttr();
      } else if (funcDef.nameIsProduct()) {
        funcDef.setAllowConstraintAttr();
        funcDef.setAllowWitnessAttr();
      }
    });
  }
  return success();
}

InFlightDiagnostic genCompareErr(StructDefOp expected, Operation *origin, const char *aspect) {
  std::string prefix = std::string();
  if (SymbolOpInterface symbol = llvm::dyn_cast<SymbolOpInterface>(origin)) {
    prefix += "\"@";
    prefix += symbol.getName();
    prefix += "\" ";
  }
  return origin->emitOpError().append(
      prefix, "must use type of its ancestor '", StructDefOp::getOperationName(), "' \"",
      expected.getHeaderString(), "\" as ", aspect, " type"
  );
}

static inline InFlightDiagnostic structFuncDefError(Operation *origin) {
  return origin->emitError() << '\'' << StructDefOp::getOperationName() << "' op "
                             << "must define either only a non-derived \"@" << FUNC_NAME_PRODUCT
                             << "\" function, or both non-derived \"@" << FUNC_NAME_COMPUTE
                             << "\" and \"@" << FUNC_NAME_CONSTRAIN << "\" functions; ";
}

/// Verifies that the given `actualType` matches the `StructDefOp` given (i.e., for the "self" type
/// parameter and return of the struct functions).
LogicalResult checkSelfType(
    SymbolTableCollection &tables, StructDefOp expectedStruct, Type actualType, Operation *origin,
    const char *aspect
) {
  if (StructType actualStructType = llvm::dyn_cast<StructType>(actualType)) {
    auto actualStructOpt =
        lookupTopLevelSymbol<StructDefOp>(tables, actualStructType.getNameRef(), origin);
    if (failed(actualStructOpt)) {
      return origin->emitError().append(
          "could not find '", StructDefOp::getOperationName(), "' named \"",
          actualStructType.getNameRef(), '"'
      );
    }
    StructDefOp actualStruct = actualStructOpt.value().get();
    if (actualStruct != expectedStruct) {
      return genCompareErr(expectedStruct, origin, aspect)
          .attachNote(actualStruct.getLoc())
          .append("uses this type instead");
    }
    // Check for an EXACT match in the parameter list since it must reference the "self" type.
    if (expectedStruct.getConstParamsAttr() != actualStructType.getParams()) {
      // To make error messages more consistent and meaningful, if the parameters don't match
      // because the actual type uses symbols that are not defined, generate an error about the
      // undefined symbol(s).
      if (ArrayAttr tyParams = actualStructType.getParams()) {
        if (failed(verifyParamsOfType(tables, tyParams.getValue(), actualStructType, origin))) {
          return failure();
        }
      }
      // Otherwise, generate an error stating the parent struct type must be used.
      return genCompareErr(expectedStruct, origin, aspect)
          .attachNote(actualStruct.getLoc())
          .append("should be type of this '", StructDefOp::getOperationName(), '\'');
    }
  } else {
    return genCompareErr(expectedStruct, origin, aspect);
  }
  return success();
}

//===------------------------------------------------------------------===//
// StructDefOp
//===------------------------------------------------------------------===//

StructType StructDefOp::getType(std::optional<ArrayAttr> constParams) {
  auto pathRes = getPathFromRoot(*this);
  assert(succeeded(pathRes)); // consistent with StructType::get() with invalid args
  return StructType::get(pathRes.value(), constParams.value_or(getConstParamsAttr()));
}

std::string StructDefOp::getHeaderString() {
  return buildStringViaCallback([this](llvm::raw_ostream &ss) {
    FailureOr<SymbolRefAttr> pathToExpected = getPathFromRoot(*this);
    if (succeeded(pathToExpected)) {
      ss << pathToExpected.value();
    } else {
      // When there is a failure trying to get the resolved name of the struct,
      //  just print its symbol name directly.
      ss << '@' << this->getSymName();
    }
    if (auto attr = this->getConstParamsAttr()) {
      ss << '<' << attr << '>';
    }
  });
}

bool StructDefOp::hasParamNamed(StringRef find) {
  if (ArrayAttr params = this->getConstParamsAttr()) {
    for (Attribute attr : params) {
      assert(llvm::isa<FlatSymbolRefAttr>(attr)); // per ODS
      if (llvm::cast<FlatSymbolRefAttr>(attr).getRootReference().strref() == find) {
        return true;
      }
    }
  }
  return false;
}

SymbolRefAttr StructDefOp::getFullyQualifiedName() {
  auto res = getPathFromRoot(*this);
  assert(succeeded(res));
  return res.value();
}

LogicalResult StructDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  if (ArrayAttr params = this->getConstParamsAttr()) {
    // Ensure struct parameter names are unique
    llvm::StringSet<> uniqNames;
    for (Attribute attr : params) {
      assert(llvm::isa<FlatSymbolRefAttr>(attr)); // per ODS
      StringRef name = llvm::cast<FlatSymbolRefAttr>(attr).getValue();
      if (!uniqNames.insert(name).second) {
        return this->emitOpError().append("has more than one parameter named \"@", name, '"');
      }
    }
    // Ensure they do not conflict with existing symbols
    for (Attribute attr : params) {
      auto res = lookupTopLevelSymbol(tables, llvm::cast<FlatSymbolRefAttr>(attr), *this, false);
      if (succeeded(res)) {
        return this->emitOpError()
            .append("parameter name \"@")
            .append(llvm::cast<FlatSymbolRefAttr>(attr).getValue())
            .append("\" conflicts with an existing symbol")
            .attachNote(res->get()->getLoc())
            .append("symbol already defined here");
      }
    }
  }
  return success();
}

namespace {

inline LogicalResult
checkMainFuncParamType(Type pType, FuncDefOp inFunc, std::optional<StructType> appendSelfType) {
  if (llvm::isa<FeltType>(pType)) {
    return success();
  } else if (auto arrayParamTy = llvm::dyn_cast<ArrayType>(pType)) {
    if (llvm::isa<FeltType>(arrayParamTy.getElementType())) {
      return success();
    }
  }

  std::string message = buildStringViaCallback([&inFunc, appendSelfType](llvm::raw_ostream &ss) {
    ss << "main entry component \"@" << inFunc.getSymName()
       << "\" function parameters must be one of: {";
    if (appendSelfType.has_value()) {
      ss << appendSelfType.value() << ", ";
    }
    ss << '!' << FeltType::name << ", ";
    ss << '!' << ArrayType::name << "<.. x !" << FeltType::name << ">}";
  });
  return inFunc.emitError(message);
}

inline LogicalResult verifyStructComputeConstrain(
    StructDefOp structDef, FuncDefOp computeFunc, FuncDefOp constrainFunc
) {
  // ASSERT: The `SetFuncAllowAttrs` trait on StructDefOp set the attributes correctly.
  assert(constrainFunc.hasAllowConstraintAttr());
  assert(!computeFunc.hasAllowConstraintAttr());
  assert(!constrainFunc.hasAllowWitnessAttr());
  assert(computeFunc.hasAllowWitnessAttr());

  // Verify parameter types are valid. Skip the first parameter of the "constrain" function; it is
  // already checked via verifyFuncTypeConstrain() in Function/IR/Ops.cpp.
  ArrayRef<Type> computeParams = computeFunc.getFunctionType().getInputs();
  ArrayRef<Type> constrainParams = constrainFunc.getFunctionType().getInputs().drop_front();
  if (structDef.isMainComponent()) {
    // Verify the input parameter types are legal. The error message is explicit about what types
    // are allowed so there is no benefit to report multiple errors if more than one parameter in
    // the referenced function has an illegal type.
    for (Type t : computeParams) {
      if (failed(checkMainFuncParamType(t, computeFunc, std::nullopt))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
    auto appendSelf = std::make_optional(structDef.getType());
    for (Type t : constrainParams) {
      if (failed(checkMainFuncParamType(t, constrainFunc, appendSelf))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
  }

  if (!typeListsUnify(computeParams, constrainParams)) {
    return constrainFunc.emitError()
        .append(
            "expected \"@", FUNC_NAME_CONSTRAIN,
            "\" function argument types (sans the first one) to match \"@", FUNC_NAME_COMPUTE,
            "\" function argument types"
        )
        .attachNote(computeFunc.getLoc())
        .append("\"@", FUNC_NAME_COMPUTE, "\" function defined here");
  }

  return success();
}

inline LogicalResult verifyStructProduct(StructDefOp structDef, FuncDefOp productFunc) {
  // ASSERT: The `SetFuncAllowAttrs` trait on StructDefOp set the attributes correctly
  assert(productFunc.hasAllowConstraintAttr());
  assert(productFunc.hasAllowWitnessAttr());

  // Verify parameter types are valid
  if (structDef.isMainComponent()) {
    ArrayRef<Type> productParams = productFunc.getFunctionType().getInputs();
    // Verify the input parameter types are legal. The error message is explicit about what types
    // are allowed so there is no benefit to report multiple errors if more than one parameter in
    // the referenced function has an illegal type.
    for (Type t : productParams) {
      if (failed(checkMainFuncParamType(t, productFunc, std::nullopt))) {
        return failure(); // checkMainFuncParamType() already emits a sufficient error message
      }
    }
  }

  return success();
}

} // namespace

LogicalResult StructDefOp::verifyRegions() {
  std::optional<FuncDefOp> foundCompute = std::nullopt;
  std::optional<FuncDefOp> foundConstrain = std::nullopt;
  std::optional<FuncDefOp> foundProduct = std::nullopt;
  {
    // Verify the following:
    // 1. The only ops within the body are member and function definitions
    // 2. The only functions defined in the struct are `@compute()` and `@constrain()`, or
    // `@product()`
    OwningEmitErrorFn emitError = getEmitOpErrFn(this);
    Region &bodyRegion = getBodyRegion();
    if (!bodyRegion.empty()) {
      for (Operation &op : bodyRegion.front()) {
        if (!llvm::isa<MemberDefOp>(op)) {
          if (FuncDefOp funcDef = llvm::dyn_cast<FuncDefOp>(op)) {
            if (funcDef.nameIsCompute()) {
              if (foundCompute) {
                return structFuncDefError(funcDef.getOperation())
                       << "found multiple \"@" << FUNC_NAME_COMPUTE << "\" functions";
              }
              foundCompute = std::make_optional(funcDef);
            } else if (funcDef.nameIsConstrain()) {
              if (foundConstrain) {
                return structFuncDefError(funcDef.getOperation())
                       << "found multiple \"@" << FUNC_NAME_CONSTRAIN << "\" functions";
              }
              foundConstrain = std::make_optional(funcDef);
            } else if (funcDef.nameIsProduct()) {
              if (foundProduct) {
                return structFuncDefError(funcDef.getOperation())
                       << "found multiple \"@" << FUNC_NAME_PRODUCT << "\" functions";
              }
              foundProduct = std::make_optional(funcDef);
            } else {
              // Must do a little more than a simple call to '?.emitOpError()' to
              // tag the error with correct location and correct op name.
              return structFuncDefError(funcDef.getOperation())
                     << "found \"@" << funcDef.getSymName() << '"';
            }
          } else {
            return op.emitOpError()
                   << "invalid operation in '" << StructDefOp::getOperationName() << "'; only '"
                   << MemberDefOp::getOperationName() << '\'' << " and '"
                   << FuncDefOp::getOperationName() << "' operations are permitted";
          }
        }
      }
    }

    if (!foundCompute.has_value() && foundConstrain.has_value()) {
      return structFuncDefError(getOperation()) << "found \"@" << FUNC_NAME_CONSTRAIN
                                                << "\", missing \"@" << FUNC_NAME_COMPUTE << "\"";
    }
    if (!foundConstrain.has_value() && foundCompute.has_value()) {
      return structFuncDefError(getOperation()) << "found \"@" << FUNC_NAME_COMPUTE
                                                << "\", missing \"@" << FUNC_NAME_CONSTRAIN << "\"";
    }
  }

  if (!foundCompute.has_value() && !foundConstrain.has_value() && !foundProduct.has_value()) {
    return structFuncDefError(getOperation())
           << "could not find \"@" << FUNC_NAME_PRODUCT << "\", \"@" << FUNC_NAME_COMPUTE
           << "\", or \"@" << FUNC_NAME_CONSTRAIN << "\"";
  }

  // Check which funcs are present and not marked with {llzk.derived}
  auto nonderived = [](std::optional<FuncDefOp> op) -> bool {
    return op && !(*op)->hasAttr(DERIVED_ATTR_NAME);
  };

  auto attachDerivedNotes = [&foundCompute, &foundConstrain,
                             &foundProduct](InFlightDiagnostic &&error) {
    if (foundProduct && (*foundProduct)->hasAttr(DERIVED_ATTR_NAME)) {
      error.attachNote(foundProduct->getLoc()) << "derived \"@" << FUNC_NAME_PRODUCT << "\" here";
    }
    if (foundCompute && (*foundCompute)->hasAttr(DERIVED_ATTR_NAME)) {
      error.attachNote(foundCompute->getLoc()) << "derived \"@" << FUNC_NAME_COMPUTE << "\" here";
    }
    if (foundConstrain && (*foundConstrain)->hasAttr(DERIVED_ATTR_NAME)) {
      error.attachNote(foundConstrain->getLoc())
          << "derived \"@" << FUNC_NAME_CONSTRAIN << "\" here";
    }
    return error;
  };

  // We know that (@compute+@constrain) is present or @product is present, or both

  // Error cases:
  // Everything is derived
  if (!nonderived(foundCompute) && !nonderived(foundConstrain) && !nonderived(foundProduct)) {
    return attachDerivedNotes(
        structFuncDefError(getOperation())
        << "could not find non-derived \"@" << FUNC_NAME_PRODUCT << "\", \"@" << FUNC_NAME_COMPUTE
        << "\", or \"@" << FUNC_NAME_CONSTRAIN << "\""
    );
  }

  // Only one of @compute/@constrain is non-derived
  if (nonderived(foundCompute) ^ nonderived(foundConstrain)) {
    return attachDerivedNotes(
        structFuncDefError(getOperation())
        << "\"@" << FUNC_NAME_COMPUTE << "\" and \"@" << FUNC_NAME_CONSTRAIN
        << "\" must both be either derived or non-derived"
    );
  }

  // Here, at least one thing is non-derived, and @compute/@constrain are derived or non-derived
  // together so everything is fine
  if (nonderived(foundCompute) && nonderived(foundConstrain) && !nonderived(foundProduct)) {
    return verifyStructComputeConstrain(*this, *foundCompute, *foundConstrain);
  }

  assert(!nonderived(foundCompute) && !nonderived(foundConstrain) && nonderived(foundProduct));
  return verifyStructProduct(*this, *foundProduct);
}

MemberDefOp StructDefOp::getMemberDef(StringAttr memberName) {
  for (Operation &op : *getBody()) {
    if (MemberDefOp memberDef = llvm::dyn_cast_if_present<MemberDefOp>(op)) {
      if (memberName.compare(memberDef.getSymNameAttr()) == 0) {
        return memberDef;
      }
    }
  }
  return nullptr;
}

std::vector<MemberDefOp> StructDefOp::getMemberDefs() {
  std::vector<MemberDefOp> res;
  for (Operation &op : *getBody()) {
    if (MemberDefOp memberDef = llvm::dyn_cast_if_present<MemberDefOp>(op)) {
      res.push_back(memberDef);
    }
  }
  return res;
}

FuncDefOp StructDefOp::getComputeFuncOp() {
  return llvm::dyn_cast_if_present<FuncDefOp>(lookupSymbol(FUNC_NAME_COMPUTE));
}

FuncDefOp StructDefOp::getConstrainFuncOp() {
  return llvm::dyn_cast_if_present<FuncDefOp>(lookupSymbol(FUNC_NAME_CONSTRAIN));
}

FuncDefOp StructDefOp::getProductFuncOp() {
  return llvm::dyn_cast_if_present<FuncDefOp>(lookupSymbol(FUNC_NAME_PRODUCT));
}

bool StructDefOp::isMainComponent() {
  FailureOr<StructType> mainTypeOpt = getMainInstanceType(this->getOperation());
  if (succeeded(mainTypeOpt)) {
    if (StructType mainType = mainTypeOpt.value()) {
      return structTypesUnify(mainType, this->getType());
    }
  }
  return false;
}

//===------------------------------------------------------------------===//
// MemberDefOp
//===------------------------------------------------------------------===//

void MemberDefOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, StringAttr sym_name, TypeAttr type,
    bool isSignal, bool isColumn
) {
  Properties &props = odsState.getOrAddProperties<Properties>();
  props.setSymName(sym_name);
  props.setType(type);
  if (isColumn) {
    props.column = odsBuilder.getUnitAttr();
  }
  if (isSignal) {
    props.signal = odsBuilder.getUnitAttr();
  }
}

void MemberDefOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, StringRef sym_name, Type type, bool isSignal,
    bool isColumn
) {
  build(
      odsBuilder, odsState, odsBuilder.getStringAttr(sym_name), TypeAttr::get(type), isSignal,
      isColumn
  );
}

void MemberDefOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attributes, bool isSignal, bool isColumn
) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
  if (isColumn) {
    odsState.getOrAddProperties<Properties>().column = odsBuilder.getUnitAttr();
  }
  if (isSignal) {
    odsState.getOrAddProperties<Properties>().signal = odsBuilder.getUnitAttr();
  }
}

void MemberDefOp::setPublicAttr(bool newValue) {
  if (newValue) {
    getOperation()->setAttr(PublicAttr::name, UnitAttr::get(getContext()));
  } else {
    getOperation()->removeAttr(PublicAttr::name);
  }
}

static LogicalResult
verifyMemberDefTypeImpl(Type memberType, SymbolTableCollection &tables, Operation *origin) {
  if (StructType memberStructType = llvm::dyn_cast<StructType>(memberType)) {
    // Special case for StructType verifies that the member type can resolve and that it is NOT the
    // parent struct (i.e., struct members cannot create circular references).
    auto memberTypeRes = verifyStructTypeResolution(tables, memberStructType, origin);
    if (failed(memberTypeRes)) {
      return failure(); // above already emits a sufficient error message
    }
    FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(origin);
    assert(succeeded(parentRes) && "MemberDefOp parent is always StructDefOp"); // per ODS def
    if (memberTypeRes.value() == parentRes.value()) {
      return origin->emitOpError()
          .append("type is circular")
          .attachNote(parentRes.value().getLoc())
          .append("references parent component defined here");
    }
    return success();
  } else {
    return verifyTypeResolution(tables, origin, memberType);
  }
}

LogicalResult MemberDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  Type memberType = this->getType();
  if (failed(verifyMemberDefTypeImpl(memberType, tables, *this))) {
    return failure();
  }

  if (!getColumn()) {
    return success();
  }
  // If the member is marked as a column only a small subset of types are allowed.
  if (!isValidColumnType(getType(), tables, *this)) {
    return emitOpError() << "marked as column can only contain felts, arrays of column types, or "
                            "structs with columns, but has type "
                         << getType();
  }
  return success();
}

//===------------------------------------------------------------------===//
// MemberRefOp implementations
//===------------------------------------------------------------------===//
namespace {

FailureOr<SymbolLookupResult<MemberDefOp>>
getMemberDefOpImpl(MemberRefOpInterface refOp, SymbolTableCollection &tables, StructType tyStruct) {
  Operation *op = refOp.getOperation();
  auto structDefRes = tyStruct.getDefinition(tables, op);
  if (failed(structDefRes)) {
    return failure(); // getDefinition() already emits a sufficient error message
  }
  // Copy namespace because we will need it later.
  llvm::SmallVector<llvm::StringRef> structDefOpNs(structDefRes->getNamespace());
  auto res = llzk::lookupSymbolIn<MemberDefOp>(
      tables, SymbolRefAttr::get(refOp->getContext(), refOp.getMemberName()),
      std::move(*structDefRes), op
  );
  if (failed(res)) {
    return refOp->emitError() << "could not find '" << MemberDefOp::getOperationName()
                              << "' named \"@" << refOp.getMemberName() << "\" in \""
                              << tyStruct.getNameRef() << '"';
  }
  // Prepend the namespace of the struct lookup since the type of the member is meant to be resolved
  // within that scope.
  res->prependNamespace(structDefOpNs);
  return std::move(res.value());
}

static FailureOr<SymbolLookupResult<MemberDefOp>>
findMember(MemberRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure the base component/struct type reference can be resolved.
  StructType tyStruct = refOp.getStructType();
  if (failed(tyStruct.verifySymbolRef(tables, refOp.getOperation()))) {
    return failure();
  }
  // Ensure the member name can be resolved in that struct.
  return getMemberDefOpImpl(refOp, tables, tyStruct);
}

static LogicalResult verifySymbolUsesImpl(
    MemberRefOpInterface refOp, SymbolTableCollection &tables,
    SymbolLookupResult<MemberDefOp> &member
) {
  // Ensure the type of the referenced member declaration matches the type used in this op.
  Type actualType = refOp.getVal().getType();
  Type memberType = member.get().getType();
  if (!typesUnify(actualType, memberType, member.getNamespace())) {
    return refOp->emitOpError() << "has wrong type; expected " << memberType << ", got "
                                << actualType;
  }
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, refOp.getOperation(), actualType);
}

LogicalResult verifySymbolUsesImpl(MemberRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure the member name can be resolved in that struct.
  auto member = findMember(refOp, tables);
  if (failed(member)) {
    return member; // getMemberDefOp() already emits a sufficient error message
  }
  return verifySymbolUsesImpl(refOp, tables, *member);
}

} // namespace

FailureOr<SymbolLookupResult<MemberDefOp>>
MemberRefOpInterface::getMemberDefOp(SymbolTableCollection &tables) {
  return getMemberDefOpImpl(*this, tables, getStructType());
}

LogicalResult MemberReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  auto member = findMember(*this, tables);
  if (failed(member)) {
    return failure();
  }
  if (failed(verifySymbolUsesImpl(*this, tables, *member))) {
    return failure();
  }
  // If the member is not a column and an offset was specified then fail to validate
  if (!member->get().getColumn() && getTableOffset().has_value()) {
    return emitOpError("cannot read with table offset from a member that is not a column")
        .attachNote(member->get().getLoc())
        .append("member defined here");
  }
  // If the member is private and this read is outside the struct, then fail to validate.
  // The current op may be inside a struct or a free function, but the
  // member op (the member definition) is always inside a struct.
  FailureOr<StructDefOp> parentRes = getParentOfType<StructDefOp>(*this);
  FailureOr<StructDefOp> memberParentRes = verifyInStruct(member->get());
  if (failed(memberParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  StructDefOp memberParentStruct = memberParentRes.value();
  if (!member->get().hasPublicAttr() &&
      (failed(parentRes) || parentRes.value() != memberParentStruct)) {
    return emitOpError()
        .append(
            "cannot read from private member of struct \"", memberParentStruct.getHeaderString(),
            "\""
        )
        .attachNote(member->get().getLoc())
        .append("member defined here");
  }
  return success();
}

LogicalResult MemberWriteOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure the write op only targets members in the current struct.
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  if (failed(checkSelfType(tables, *getParentRes, getComponent().getType(), *this, "base value"))) {
    return failure(); // checkSelfType() already emits a sufficient error message
  }
  // Perform the standard member ref checks.
  return verifySymbolUsesImpl(*this, tables);
}

//===------------------------------------------------------------------===//
// MemberReadOp
//===------------------------------------------------------------------===//

void MemberReadOp::build(
    OpBuilder &builder, OperationState &state, Type resultType, Value component, StringAttr member
) {
  Properties &props = state.getOrAddProperties<Properties>();
  props.setMemberName(FlatSymbolRefAttr::get(member));
  state.addTypes(resultType);
  state.addOperands(component);
  affineMapHelpers::buildInstantiationAttrsEmptyNoSegments<MemberReadOp>(builder, state);
}

void MemberReadOp::build(
    OpBuilder &builder, OperationState &state, Type resultType, Value component, StringAttr member,
    Attribute dist, ValueRange mapOperands, std::optional<int32_t> numDims
) {
  // '!mapOperands.empty()' implies 'numDims.has_value()'
  assert(mapOperands.empty() || numDims.has_value());
  state.addOperands(component);
  state.addTypes(resultType);
  if (numDims.has_value()) {
    affineMapHelpers::buildInstantiationAttrsNoSegments<MemberReadOp>(
        builder, state, ArrayRef({mapOperands}), builder.getDenseI32ArrayAttr({*numDims})
    );
  } else {
    affineMapHelpers::buildInstantiationAttrsEmptyNoSegments<MemberReadOp>(builder, state);
  }
  Properties &props = state.getOrAddProperties<Properties>();
  props.setMemberName(FlatSymbolRefAttr::get(member));
  props.setTableOffset(dist);
}

void MemberReadOp::build(
    OpBuilder & /*odsBuilder*/, OperationState &odsState, TypeRange resultTypes,
    ValueRange operands, ArrayRef<NamedAttribute> attrs
) {
  odsState.addTypes(resultTypes);
  odsState.addOperands(operands);
  odsState.addAttributes(attrs);
}

LogicalResult MemberReadOp::verify() {
  SmallVector<AffineMapAttr, 1> mapAttrs;
  if (AffineMapAttr map =
          llvm::dyn_cast_if_present<AffineMapAttr>(getTableOffset().value_or(nullptr))) {
    mapAttrs.push_back(map);
  }
  return affineMapHelpers::verifyAffineMapInstantiations(
      getMapOperands(), getNumDimsPerMap(), mapAttrs, *this
  );
}

//===------------------------------------------------------------------===//
// CreateStructOp
//===------------------------------------------------------------------===//

void CreateStructOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  setNameFn(getResult(), "self");
}

LogicalResult CreateStructOp::verifySymbolUses(SymbolTableCollection &tables) {
  FailureOr<StructDefOp> getParentRes = verifyInStruct(*this);
  if (failed(getParentRes)) {
    return failure(); // verifyInStruct() already emits a sufficient error message
  }
  if (failed(checkSelfType(tables, *getParentRes, this->getType(), *this, "result"))) {
    return failure();
  }
  return success();
}

} // namespace llzk::component
