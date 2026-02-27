//===-- Ops.cpp - Global value operation implementations --------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Types.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
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

//===------------------------------------------------------------------===//
// GlobalDefOp
//===------------------------------------------------------------------===//

ParseResult GlobalDefOp::parseGlobalInitialValue(
    OpAsmParser &parser, Attribute &initialValue, TypeAttr typeAttr
) {
  if (parser.parseOptionalEqual()) {
    // When there's no equal sign, there's no initial value to parse.
    return success();
  }
  Type specifiedType = typeAttr.getValue();

  // Special case for parsing LLZK FeltType to match format of FeltConstantOp.
  // Not actually necessary but the default format is verbose. ex: "#llzk<felt.const 35>"
  if (isa<FeltType>(specifiedType)) {
    FeltConstAttr feltConstAttr;
    if (parser.parseCustomAttributeWithFallback<FeltConstAttr>(feltConstAttr)) {
      return failure();
    }
    initialValue = feltConstAttr;
    return success();
  }
  // Fallback to default parser for all other types.
  if (failed(parser.parseAttribute(initialValue, specifiedType))) {
    return failure();
  }
  return success();
}

void GlobalDefOp::printGlobalInitialValue(
    OpAsmPrinter &p, GlobalDefOp /*op*/, Attribute initialValue, TypeAttr /*typeAttr*/
) {
  if (initialValue) {
    p << " = ";
    // Special case for LLZK FeltType to match format of FeltConstantOp.
    // Not actually necessary but the default format is verbose. ex: "#llzk<felt.const 35>"
    if (FeltConstAttr feltConstAttr = llvm::dyn_cast<FeltConstAttr>(initialValue)) {
      p.printStrippedAttrOrType<FeltConstAttr>(feltConstAttr);
    } else {
      p.printAttributeWithoutType(initialValue);
    }
  }
}

LogicalResult GlobalDefOp::verifySymbolUses(SymbolTableCollection &tables) {
  // Ensure any SymbolRef used in the type are valid
  return verifyTypeResolution(tables, *this, getType());
}

namespace {

inline InFlightDiagnosticWrapper reportMismatch(
    EmitErrorFn errFn, Type rootType, const Twine &aspect, const Twine &expected, const Twine &found
) {
  return errFn().append(
      "with type ", rootType, " expected ", expected, " ", aspect, " but found ", found
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
    if (!llvm::isa<FeltConstAttr, IntegerAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "felt.type", valAttr);
    }
  } else if (llvm::isa<StringType>(type)) {
    if (!llvm::isa<StringAttr>(valAttr)) {
      return reportMismatch(errFn, rootType, aspect, "builtin.string", valAttr);
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
