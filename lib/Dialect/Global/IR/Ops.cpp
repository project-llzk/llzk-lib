//===-- Ops.cpp - Global value operation implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Felt/IR/Ops.h"

#include "InitializerUtils.h"

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/String/IR/Types.h"
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

static inline FailureOr<NormalizedGlobalInitializer>
normalizeGlobalInitializer(IndexType expectedType, Attribute value, EmitErrorFn emitError) {
  if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
    if (!llvm::isa<BoolAttr>(value)) {
      APInt intValueBits = intValue.getValue();
      if (intValue.getType().isSignlessInteger() && intValueBits.isNegative() &&
          intValueBits.getBitWidth() < IndexType::kInternalStorageBitWidth) {
        return emitError().append(
            "signless integer with sign bit set cannot be widened to `index`"
        );
      }
      FailureOr<IntegerAttr> normalized = forceIntType(intValue, emitError);
      if (failed(normalized)) {
        return failure();
      }
      value = *normalized;
    }
  }
  return NormalizedGlobalInitializer {expectedType, value};
}

static inline FailureOr<NormalizedGlobalInitializer>
normalizeGlobalInitializer(FeltType expectedType, Attribute value, EmitErrorFn) {
  if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
    value = FeltConstAttr::get(value.getContext(), intValue.getValue(), expectedType);
  } else if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(value)) {
    FeltType valueType = feltValue.getType();
    if (!expectedType.hasField() && valueType.hasField()) {
      expectedType = valueType;
    } else if (expectedType.hasField() && !valueType.hasField()) {
      value = FeltConstAttr::get(value.getContext(), feltValue.getValue(), expectedType);
    }
  }
  return NormalizedGlobalInitializer {expectedType, value};
}

static inline FailureOr<NormalizedGlobalInitializer>
normalizeGlobalInitializer(ArrayType expectedType, Attribute value, EmitErrorFn emitError) {
  if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value)) {
    Type elementType = expectedType.getElementType();
    if (auto feltElementType = llvm::dyn_cast<FeltType>(elementType)) {
      // Infer an omitted field from a typed felt element before normalizing the
      // full array. A conflicting explicit field is left for verification so it
      // can report the conflict against the original initializer.
      for (Attribute element : arrayValue) {
        if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(element)) {
          FeltType valueType = feltValue.getType();
          if (!valueType.hasField()) {
            continue;
          }
          if (feltElementType.hasField() && feltElementType != valueType) {
            return NormalizedGlobalInitializer {expectedType, value};
          }
          feltElementType = valueType;
        }
      }
      elementType = feltElementType;
      expectedType = expectedType.cloneWith(elementType);
    }

    // Normalize each element recursively. This also lets a nested initializer
    // refine the element type (for example, an unqualified felt type).
    SmallVector<Attribute> elements;
    elements.reserve(arrayValue.size());
    for (Attribute element : arrayValue) {
      FailureOr<NormalizedGlobalInitializer> normalized =
          llzk::global::normalizeGlobalInitializer(elementType, element, emitError);
      if (failed(normalized)) {
        return failure();
      }
      elementType = normalized->type;
      elements.push_back(normalized->value);
    }
    // Rebuild the array type and attribute from the normalized element data.
    expectedType = expectedType.cloneWith(elementType);
    value = ArrayAttr::get(value.getContext(), elements);
  }
  return NormalizedGlobalInitializer {expectedType, value};
}

static inline FailureOr<NormalizedGlobalInitializer>
normalizeGlobalInitializer(StringType expectedType, Attribute value, EmitErrorFn) {
  if (auto stringValue = llvm::dyn_cast<StringAttr>(value)) {
    value = StringAttr::get(stringValue.getValue(), expectedType);
  }
  return NormalizedGlobalInitializer {expectedType, value};
}

} // namespace

FailureOr<NormalizedGlobalInitializer>
normalizeGlobalInitializer(Type expectedType, Attribute value, EmitErrorFn emitError) {
  if (auto idxType = llvm::dyn_cast<IndexType>(expectedType)) {
    return normalizeGlobalInitializer(idxType, value, emitError);
  } else if (auto feltType = llvm::dyn_cast<FeltType>(expectedType)) {
    return normalizeGlobalInitializer(feltType, value, emitError);
  } else if (auto arrayType = llvm::dyn_cast<ArrayType>(expectedType)) {
    return normalizeGlobalInitializer(arrayType, value, emitError);
  } else if (auto stringType = llvm::dyn_cast<StringType>(expectedType)) {
    return normalizeGlobalInitializer(stringType, value, emitError);
  } else if (expectedType.isSignlessInteger(1)) {
    if (auto intValue = llvm::dyn_cast<IntegerAttr>(value)) {
      APInt intValueBits = intValue.getValue();
      if (!intValueBits.isZero() && !intValueBits.isOne()) {
        return emitError().append("integer constant out of range for attribute");
      }
      value = IntegerAttr::get(expectedType, APInt(1, intValueBits.getZExtValue()));
    }
    return NormalizedGlobalInitializer {expectedType, value};
  } else {
    return NormalizedGlobalInitializer {expectedType, value};
  }
}

//===------------------------------------------------------------------===//
// GlobalDefOp
//===------------------------------------------------------------------===//

static ParseResult normalizeParsedInitialValue(
    OpAsmParser &parser, SMLoc initializerLoc, Type &declaredType, Attribute &initialValue
) {
  FailureOr<NormalizedGlobalInitializer> normalized =
      normalizeGlobalInitializer(declaredType, initialValue, [&parser, initializerLoc] {
    return InFlightDiagnosticWrapper(parser.emitError(initializerLoc));
  });
  if (failed(normalized)) {
    return failure();
  }
  declaredType = normalized->type;
  initialValue = normalized->value;
  return success();
}

/// Parse an initializer attribute recursively so felt values retain their optional field syntax
/// even when nested in an array. If there is a conflict among the felt values, the verifier will
/// catch it later.
static ParseResult parseInitialValueForType(OpAsmParser &parser, Type type, Attribute &value) {
  if (llvm::isa<FeltType>(type)) {
    FeltConstAttr feltValue;
    if (parser.parseCustomAttributeWithFallback<FeltConstAttr>(feltValue)) {
      return failure();
    }
    value = feltValue;
    return success();
  }
  if (auto arrayType = llvm::dyn_cast<ArrayType>(type);
      arrayType && llvm::isa<FeltType>(arrayType.getElementType())) {
    SmallVector<Attribute> elements;
    auto parseElement = [&]() -> ParseResult {
      Attribute element;
      if (failed(parseInitialValueForType(parser, arrayType.getElementType(), element))) {
        return failure();
      }
      elements.push_back(element);
      return success();
    };
    if (failed(parser.parseCommaSeparatedList(AsmParser::Delimiter::Square, parseElement))) {
      return failure();
    }
    value = ArrayAttr::get(parser.getContext(), elements);
    return success();
  }
  return parser.parseAttribute(value, type);
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
  if (succeeded(parser.parseOptionalEqual())) {
    Attribute initialValue;
    SMLoc initializerLoc = parser.getCurrentLocation();
    if (failed(parseInitialValueForType(parser, declaredType, initialValue)) ||
        failed(normalizeParsedInitialValue(parser, initializerLoc, declaredType, initialValue))) {
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

namespace {

/// Print an initializer recursively without its redundant storage types.
/// GlobalDefOp's declared type supplies the type for every initializer value.
static void printInitialValue(AsmPrinter &printer, Attribute value) {
  if (auto arrayValue = llvm::dyn_cast<ArrayAttr>(value)) {
    printer << '[';
    llvm::interleaveComma(arrayValue, printer.getStream(), [&printer](Attribute element) {
      printInitialValue(printer, element);
    });
    printer << ']';
  } else if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(value)) {
    printer.printStrippedAttrOrType<FeltConstAttr>(feltValue);
  } else {
    printer.printAttributeWithoutType(value);
  }
}

} // namespace

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
  return verifyTypeResolution(tables, *this, getType());
}

namespace {

static inline InFlightDiagnosticWrapper reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, const Twine &found
) {
  return errFn().append(
      "with type ", rootType, " expected ", expected, ' ', aspect, " but found ", found
  );
}

static inline InFlightDiagnosticWrapper reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, Attribute found
) {
  return reportMismatch(errFn, rootType, aspect, expected, found.getAbstractAttribute().getName());
}

static LogicalResult ensureAttrTypeMatch(
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
      if (auto feltElemTy = llvm::dyn_cast<FeltType>(arrTy.getElementType())) {
        for (Attribute element : arrVal) {
          if (auto feltValue = llvm::dyn_cast<FeltConstAttr>(element)) {
            FeltType valueType = feltValue.getType();
            if (!valueType.hasField()) {
              continue;
            }
            if (feltElemTy.hasField() && feltElemTy != valueType) {
              return errFn().append(
                  "initializer array contains conflicting types ", valueType, " vs ", feltElemTy
              );
            }
            feltElemTy = valueType;
          }
        }
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

static FailureOr<SymbolLookupResult<GlobalDefOp>>
verifySymbolUsesImpl(GlobalRefOpInterface refOp, SymbolTableCollection &tables) {
  // Ensure this op references a valid GlobalDefOp name
  auto tgt = refOp.getGlobalDefOp(tables);
  if (failed(tgt)) {
    return failure();
  }
  // Ensure the SSA Value type matches the GlobalDefOp type
  Type globalType = tgt->get().getType();
  if (!typesUnify(refOp.getVal().getType(), globalType, tgt->getIncludeSymNames())) {
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
