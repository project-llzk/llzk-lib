//===-- Ops.cpp - Global value operation implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Ops.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Types.h"
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Types.h"
#include "llzk/Util/BuilderHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

// TableGen'd implementation files
#include "llzk/Dialect/Global/IR/OpInterfaces.cpp.inc"

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Global/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::array;
using namespace llzk::felt;
using namespace llzk::string;

namespace llzk::global {

namespace {

/// Print an initializer recursively without its redundant storage types.
/// GlobalDefOp's declared type supplies the type for scalar initializer values.
/// Array elements use generic attribute syntax, which requires felt attributes
/// to retain their dialect mnemonic and type.
void printInitialValue(AsmPrinter &printer, Attribute value, bool isArrayElement = false) {
  if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value)) {
    printer << '[';
    llvm::interleaveComma(arrayValue, printer.getStream(), [&printer](Attribute element) {
      printInitialValue(printer, element, true);
    });
    printer << ']';
  } else if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(value)) {
    if (isArrayElement) {
      printer.printAttributeWithoutType(feltValue);
    } else {
      printer.printStrippedAttrOrType<FeltConstAttr>(feltValue);
    }
  } else {
    printer.printAttributeWithoutType(value);
  }
}

/// Returns one global definition that coordinates refinement verification for the root.
///
/// Symbol-use verification invokes every global definition independently. Selecting a
/// coordinator lets us collect all global refinements in one root walk instead of repeating that
/// walk for each definition. Globals are normally direct children of the root module. The walk is
/// a fallback for roots that contain globals only in nested modules.
GlobalDefOp getRefinementVerificationCoordinator(ModuleOp root) {
  for (Operation &op : root.getBody()->getOperations()) {
    if (auto global = llvm::dyn_cast<GlobalDefOp>(op)) {
      return global;
    }
  }

  GlobalDefOp coordinator;
  root.walk([&coordinator](GlobalDefOp global) {
    coordinator = global;
    return WalkResult::interrupt();
  });
  return coordinator;
}

/// Verify that every felt position of each mutable global is refined to one field.
///
/// This performs one root walk to collect mutable globals with felt positions and one to process
/// global references. In addition to fields written on the reference itself, reads collect the
/// field expectations of their direct call consumers. Each reference is resolved at most once,
/// independent of the number of mutable globals.
LogicalResult verifyGlobalFeltRefinements(ModuleOp root, SymbolTableCollection &tables) {
  SmallVector<FeltRefinement> refinements;
  root.walk([&refinements, &tables](GlobalRefOpInterface refOp) {
    // This scan is auxiliary to each reference's own verifier, which reports
    // lookup failures. Avoid emitting duplicate diagnostics for unresolved
    // references while collecting refinements.
    auto target = lookupTopLevelSymbol<GlobalDefOp>(
        tables, refOp.getNameRef(), refOp.getOperation(), /*reportMissing=*/false
    );
    if (failed(target)) {
      return WalkResult::advance();
    }
    GlobalDefOp global = target->get();
    if (global.isConstant()) {
      return WalkResult::advance();
    }
    SmallVector<FeltType> globalFeltTypes;
    collectFeltTypes(global.getType(), globalFeltTypes);
    if (globalFeltTypes.empty()) {
      return WalkResult::advance();
    }
    // Only unifiable reference types have felt positions corresponding to
    // this global. A non-unifying reference is diagnosed by its own symbol
    // verifier, so it must not participate in refinement collection.
    if (!typesUnify(refOp.getVal().getType(), global.getType(), target->getIncludeSymNames())) {
      return WalkResult::advance();
    }
    SmallVector<std::pair<Operation *, Type>> refinementTypes;
    refinementTypes.emplace_back(refOp.getOperation(), refOp.getVal().getType());
    if (auto readOp = llvm::dyn_cast<GlobalReadOp>(refOp.getOperation())) {
      for (OpOperand &use : readOp.getVal().getUses()) {
        auto callOp = llvm::dyn_cast<function::CallOp>(use.getOwner());
        unsigned argIndex = use.getOperandNumber();
        if (!callOp || argIndex >= callOp.getArgOperands().size()) {
          continue;
        }
        // A malformed or unresolved call is diagnosed by its own verifier. Do
        // not emit an additional error while collecting its field refinement.
        auto callee = callOp.getCalleeTarget(tables);
        if (failed(callee)) {
          continue;
        }
        refinementTypes.emplace_back(
            callOp.getOperation(), callee->get().getFunctionType().getInput(argIndex)
        );
      }
    }
    for (auto [origin, refinementType] : refinementTypes) {
      // The referenced value is only a field refinement of this global when
      // it can represent the global's storage type. Other type errors are
      // reported by the reference or call verifier that owns the use.
      if (!typesUnify(refinementType, global.getType(), target->getIncludeSymNames())) {
        continue;
      }
      refinements.push_back(
          {global.getOperation(), global.getType(), origin, refinementType, "global",
           global.getSymName()}
      );
    }
    return WalkResult::advance();
  });
  return verifyFeltRefinements(refinements);
}

} // namespace

//===------------------------------------------------------------------===//
// GlobalDefOp
//===------------------------------------------------------------------===//

/// Resolve a parsed initializer's felt fields into its declared global type.
///
/// A field-qualified value refines an unspecified felt declaration. Conversely,
/// an unqualified value adopts an explicitly declared field. The resulting type
/// and attribute therefore always agree exactly.
static ParseResult normalizeParsedInitialValue(
    OpAsmParser &parser, SMLoc initializerLoc, Type &declaredType, Attribute &initialValue
) {
  auto normalizeFelt = [&](FeltType declaredFelt, Attribute value,
                           FeltType &resolvedFelt) -> FailureOr<FeltConstAttr> {
    if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(value)) {
      FeltType valueType = feltValue.getType();
      if (declaredFelt.hasField() && valueType.hasField() && declaredFelt != valueType) {
        return parser.emitError(initializerLoc) << "initializer type " << valueType
                                                << " conflicts with declared type " << declaredFelt;
      }
      resolvedFelt = declaredFelt.hasField() ? declaredFelt : valueType;
      return FeltConstAttr::get(parser.getContext(), feltValue.getValue(), resolvedFelt);
    }
    if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
      resolvedFelt = declaredFelt;
      return FeltConstAttr::get(parser.getContext(), intValue.getValue(), resolvedFelt);
    }
    return parser.emitError(initializerLoc) << "expected a felt initializer value";
  };

  if (auto feltType = llvm::dyn_cast<FeltType>(declaredType)) {
    FeltType resolvedFelt;
    FailureOr<FeltConstAttr> normalized = normalizeFelt(feltType, initialValue, resolvedFelt);
    if (failed(normalized)) {
      return failure();
    }
    declaredType = resolvedFelt;
    initialValue = *normalized;
    return success();
  }

  auto arrayType = llvm::dyn_cast<ArrayType>(declaredType);
  auto arrayValue = llvm::dyn_cast<ArrayAttr>(initialValue);
  if (!arrayType || !arrayValue) {
    return success();
  }
  auto elementFeltType = llvm::dyn_cast<FeltType>(arrayType.getElementType());
  if (elementFeltType) {
    FeltType resolvedElementType = elementFeltType;
    for (Attribute element : arrayValue) {
      if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(element)) {
        auto feltValueType = feltValue.getType();
        if (feltValueType.hasField()) {
          if (resolvedElementType != feltValueType && resolvedElementType.hasField()) {
            return parser.emitError(initializerLoc)
                   << "initializer array contains conflicting types " << feltValueType << " vs "
                   << resolvedElementType;
          }
          resolvedElementType = feltValueType;
        }
      }
    }

    SmallVector<Attribute> normalizedElements;
    normalizedElements.reserve(arrayValue.size());
    for (Attribute element : arrayValue) {
      FeltType unused;
      FailureOr<FeltConstAttr> normalized = normalizeFelt(resolvedElementType, element, unused);
      if (failed(normalized)) {
        return failure();
      }
      normalizedElements.push_back(*normalized);
    }
    declaredType = arrayType.cloneWith(resolvedElementType);
    initialValue = ArrayAttr::get(parser.getContext(), normalizedElements);
    return success();
  }

  Type elementType = arrayType.getElementType();
  if (elementType.isSignlessInteger(1)) {
    SmallVector<Attribute> normalizedElements;
    normalizedElements.reserve(arrayValue.size());
    for (Attribute element : arrayValue) {
      if (auto intValue = llvm::dyn_cast<IntegerAttr>(element)) {
        APInt value = intValue.getValue();
        if (!value.isZero() && !value.isOne()) {
          return parser.emitError(initializerLoc) << "integer constant out of range for attribute";
        }
        normalizedElements.push_back(IntegerAttr::get(elementType, value.trunc(1)));
      } else {
        normalizedElements.push_back(element);
      }
    }
    initialValue = ArrayAttr::get(parser.getContext(), normalizedElements);
  } else if (llvm::isa<IndexType>(elementType)) {
    SmallVector<Attribute> normalizedElements;
    normalizedElements.reserve(arrayValue.size());
    for (Attribute element : arrayValue) {
      if (llvm::isa<BoolAttr>(element)) {
        // BoolAttr is an IntegerAttr subtype, but index initializers must not
        // accept boolean values. Preserve it for the verifier to reject.
        normalizedElements.push_back(element);
      } else if (auto intValue = llvm::dyn_cast<IntegerAttr>(element)) {
        auto emitError = [&parser, initializerLoc] {
          return InFlightDiagnosticWrapper(parser.emitError(initializerLoc));
        };
        FailureOr<IntegerAttr> normalized = forceIntType(intValue, emitError);
        if (failed(normalized)) {
          return failure();
        }
        normalizedElements.push_back(*normalized);
      } else {
        normalizedElements.push_back(element);
      }
    }
    initialValue = ArrayAttr::get(parser.getContext(), normalizedElements);
  } else if (auto stringType = llvm::dyn_cast<StringType>(elementType)) {
    SmallVector<Attribute> normalizedElements;
    normalizedElements.reserve(arrayValue.size());
    for (Attribute element : arrayValue) {
      if (auto stringValue = llvm::dyn_cast<StringAttr>(element)) {
        normalizedElements.push_back(StringAttr::get(stringValue.getValue(), stringType));
      } else {
        normalizedElements.push_back(element);
      }
    }
    initialValue = ArrayAttr::get(parser.getContext(), normalizedElements);
  }
  return success();
}

ParseResult GlobalDefOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &props = result.getOrAddProperties<GlobalDefOp::Properties>();
  if (succeeded(parser.parseOptionalKeyword("const"))) {
    props.constant = parser.getBuilder().getUnitAttr();
  }

  StringAttr symName;
  if (parser.parseSymbolName(symName) || parser.parseColon()) {
    return failure();
  }
  props.sym_name = symName;

  TypeAttr typeAttr;
  if (parser.parseCustomAttributeWithFallback(typeAttr, parser.getBuilder().getNoneType())) {
    return failure();
  }
  Type declaredType = typeAttr.getValue();

  Attribute initialValue;
  if (succeeded(parser.parseOptionalEqual())) {
    SMLoc initializerLoc = parser.getCurrentLocation();
    if (llvm::isa<FeltType>(declaredType)) {
      FeltConstAttr feltValue;
      if (parser.parseCustomAttributeWithFallback<FeltConstAttr>(feltValue)) {
        return failure();
      }
      initialValue = feltValue;
    } else if (failed(parser.parseAttribute(initialValue, declaredType))) {
      return failure();
    }
    if (failed(normalizeParsedInitialValue(parser, initializerLoc, declaredType, initialValue))) {
      return failure();
    }
    props.initial_value = initialValue;
  }
  props.type = TypeAttr::get(declaredType);

  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return verifyInherentAttrs(result.name, result.attributes, [&]() {
    return parser.emitError(loc) << '\'' << result.name.getStringRef() << "' op ";
  });
}

void GlobalDefOp::print(OpAsmPrinter &p) {
  if (getConstant()) {
    p << " const";
  }
  p << ' ';
  p.printSymbolName(getSymName());
  p << " : ";
  p.printAttributeWithoutType(getTypeAttr());
  if (Attribute initialValue = getInitialValueAttr()) {
    p << " = ";
    printInitialValue(p, initialValue);
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"constant", "sym_name", "type", "initial_value"});
}

LogicalResult GlobalDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  if (failed(verifyTypeResolution(tables, *this, getType()))) {
    return failure();
  }

  auto root = getTopRootModule(getOperation());
  if (failed(root)) {
    return failure();
  }
  if (getRefinementVerificationCoordinator(*root) != *this) {
    return success();
  }
  return verifyGlobalFeltRefinements(*root, tables);
}

namespace {

inline InFlightDiagnosticWrapper reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, const Twine &found
) {
  return errFn().append(
      "with type ", rootType, " expected ", expected, ' ', aspect, " but found ", found
  );
}

inline InFlightDiagnosticWrapper reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, Attribute found
) {
  return reportMismatch(errFn, rootType, aspect, expected, found.getAbstractAttribute().getName());
}

LogicalResult ensureAttrTypeMatch(
    Type type, Attribute valAttr, const OwningEmitErrorFn &errFn, Type rootType, const Twine &aspect
) {
  if (!isValidGlobalType(type)) {
    // Same error message ODS-generated code would produce
    return errFn().append(
        "attribute 'type' failed to satisfy constraint: type attribute of "
        "any LLZK type except non-constant types"
    );
  }
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(valAttr);
      typedAttr && typedAttr.getType() != type) {
    return errFn().append(
        "with type ", rootType, " expected ", aspect, " with type ", type, " but found ",
        typedAttr.getType()
    );
  }
  if (type.isSignlessInteger(1)) {
    if (IntegerAttr ia = llvm::dyn_cast<IntegerAttr>(valAttr)) {
      APInt val = ia.getValue();
      if (!val.isZero() && !val.isOne()) {
        return errFn().append("integer constant out of range for attribute");
      }
    } else if (!llvm::isa<BoolAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "builtin.bool or builtin.integer", valAttr);
    }
  } else if (llvm::isa<IndexType>(type)) {
    // The explicit check for BoolAttr is needed because the LLVM isa/cast functions treat
    // BoolAttr as a subtype of IntegerAttr but this scenario should not allow BoolAttr.
    bool isBool = llvm::isa<BoolAttr>(valAttr);
    if (isBool || !llvm::isa<IntegerAttr>(valAttr)) {
      return reportMismatch(
          errFn, rootType, aspect, "builtin.index",
          isBool ? "builtin.bool" : valAttr.getAbstractAttribute().getName()
      );
    }
  } else if (llvm::isa<FeltType>(type)) {
    if (!llvm::isa<FeltConstAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "felt.type", valAttr);
    }
  } else if (llvm::isa<StringType>(type)) {
    if (!llvm::isa<StringAttr>(valAttr)) {
      return errFn().append(
          "with type ", rootType, " expected ", aspect, " with type ", type, " but found ",
          valAttr.getAbstractAttribute().getName()
      );
    }
  } else if (ArrayType arrTy = llvm::dyn_cast<ArrayType>(type)) {
    if (ArrayAttr arrVal = llvm::dyn_cast<ArrayAttr>(valAttr)) {
      // Ensure the number of elements is correct for the ArrayType
      assert(arrTy.hasStaticShape() && "implied by earlier isValidGlobalType() check");
      int64_t expectedCount = arrTy.getNumElements();
      size_t actualCount = arrVal.size();
      if (std::cmp_not_equal(actualCount, expectedCount)) {
        return reportMismatch(
            errFn, rootType, Twine(aspect) + " to contain " + Twine(expectedCount) + " elements",
            "builtin.array", Twine(actualCount)
        );
      }
      // Ensure the type of each element is correct for the ArrayType.
      // Rather than immediately returning on failure, check all elements and aggregate to provide
      // as many errors are possible in a single verifier run.
      bool hasFailure = false;
      Type expectedElemTy = arrTy.getElementType();
      for (Attribute e : arrVal.getValue()) {
        hasFailure |=
            failed(ensureAttrTypeMatch(expectedElemTy, e, errFn, rootType, "array element"));
      }
      if (hasFailure) {
        return failure();
      }
    } else {
      return reportMismatch(errFn, rootType, aspect, "builtin.array", valAttr);
    }
  } else {
    return errFn().append("expected a valid LLZK type but found ", type);
  }
  return success();
}

} // namespace

LogicalResult GlobalDefOp::verify() {
  if (Attribute initValAttr = getInitialValueAttr()) {
    Type ty = getType();
    OwningEmitErrorFn errFn = getEmitOpErrFn(this);
    return ensureAttrTypeMatch(ty, initValAttr, errFn, ty, "attribute value");
  }
  // If there is no initial value, it cannot have "const".
  if (isConstant()) {
    return emitOpError("marked as 'const' must be assigned a value");
  }
  return success();
}

//===------------------------------------------------------------------===//
// GlobalReadOp / GlobalWriteOp
//===------------------------------------------------------------------===//

FailureOr<SymbolLookupResult<GlobalDefOp>>
GlobalRefOpInterface::getGlobalDefOp(SymbolTableCollection &tables) {
  return lookupTopLevelSymbol<GlobalDefOp>(tables, getNameRef(), getOperation());
}

namespace {

FailureOr<SymbolLookupResult<GlobalDefOp>>
verifySymbolUsesImpl(GlobalRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure this op references a valid GlobalDefOp name
  auto tgt = refOp.getGlobalDefOp(tables);
  if (failed(tgt)) {
    return failure();
  }
  // Ensure the SSA Value type matches the GlobalDefOp type
  Type globalType = tgt->get().getType();
  if (!typesUnifyWithoutLosingFeltFields(
          globalType, refOp.getVal().getType(), tgt->getIncludeSymNames()
      )) {
    return refOp->emitOpError() << "has wrong type; expected " << globalType << ", got "
                                << refOp.getVal().getType();
  }
  return tgt;
}

} // namespace

LogicalResult GlobalReadOp::verifySymbolUses(SymbolTableCollection &tables) {
  if (failed(verifySymbolUsesImpl(*this, tables))) {
    return failure();
  }
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getType());
}

LogicalResult GlobalWriteOp::verifySymbolUses(SymbolTableCollection &tables) {
  auto tgt = verifySymbolUsesImpl(*this, tables);
  if (failed(tgt)) {
    return failure();
  }
  if (tgt->get().isConstant()) {
    return emitOpError().append(
        "cannot target '", GlobalDefOp::getOperationName(), "' marked as 'const'"
    );
  }
  return success();
}

} // namespace llzk::global
