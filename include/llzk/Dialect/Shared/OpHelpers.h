//===-- OpHelpers.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/LLZK/IR/AttributeHelper.h"
#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/ErrorHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>

namespace llzk {

/// Get the operation name, like "constrain.eq" for the given OpClass.
/// This function can be used when the compiler would complain about
/// incomplete types if `OpClass::getOperationName()` were called directly.
template <typename OpClass> inline llvm::StringLiteral getOperationName() {
  return OpClass::getOperationName();
}

/// Return the closest surrounding parent/ancestor operation that is of type 'OpClass',
/// either the op itself or an ancestor.
template <typename OpClass> inline OpClass getSelfOrParentOfType(mlir::Operation *op) {
  if (op) {
    if (OpClass self = llvm::dyn_cast<OpClass>(op)) {
      return self;
    }
    if (OpClass parent = op->getParentOfType<OpClass>()) {
      return parent;
    }
  }
  return {};
}

/// Return the closest surrounding parent/ancestor operation that is of type 'OpClass'.
template <typename OpClass> inline OpClass getParentOfType(mlir::Operation *op) {
  if (op) {
    if (OpClass p = op->getParentOfType<OpClass>()) {
      return p;
    }
  }
  return {};
}

/// Return true if the parameter has a parent/ancestor op that is an instance of one
/// of the template type arguments.
template <typename... OpTys> bool hasParentThatIsa(mlir::Operation *op) {
  while ((op = op->getParentOp())) {
    if (llvm::isa<OpTys...>(op)) {
      return true;
    }
  }
  return false;
}

/// See `LLZKSymbolTable` ODS documentation for details.
template <typename TypeClass>
// Suppress false positive from `clang-tidy`
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
class LLZKSymbolTableImplTrait
    : public mlir::OpTrait::TraitBase<TypeClass, LLZKSymbolTableImplTrait> {
public:
  static mlir::LogicalResult verifyRegionTrait(mlir::Operation *op) {
    // Note: the current op will be checked by the normal `SymbolTable` trait that is
    // included in `LLZKSymbolTable`. Checking it here would cause the same error described
    // in `LLZKSymbolTable`.
    while ((op = op->getParentWithTrait<mlir::OpTrait::SymbolTable>())) {
      if (mlir::failed(mlir::detail::verifySymbolTable(op))) {
        return mlir::failure();
      }
    }
    return mlir::success();
  }
};

/// See `HasAncestor` ODS documentation for details.
template <typename... Ancestors> struct HasAncestor {
  template <typename ConcreteType>
  static void appendTypeName(mlir::InFlightDiagnostic &diag, bool &first) {
    if (!first) {
      diag << ", ";
    }
    first = false;
    diag << '\'' << ConcreteType::getOperationName() << '\'';
  }

  template <typename ConcreteType>
  // Suppress false positive from `clang-tidy`
  // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
  struct Impl : public mlir::OpTrait::TraitBase<ConcreteType, Impl> {
    static mlir::LogicalResult verifyRegionTrait(mlir::Operation *op) {
      if (hasParentThatIsa<Ancestors...>(op)) {
        return mlir::success();
      }
      auto diag = op->emitOpError();

      if constexpr (sizeof...(Ancestors) == 1) {
        diag << "must have an ancestor of type ";
      } else {
        diag << "must have an ancestor of one of the following types: ";
      }

      bool first = true;
      (HasAncestor::template appendTypeName<Ancestors>(diag, first), ...);

      return diag;
    }
  };
};

/// Produces errors if there is an inconsistency in the various attributes/values that are used to
/// support affine map instantiation in the Op marked with this Trait.
template <int OperandSegmentIndex> struct VerifySizesForMultiAffineOps {
  template <typename TypeClass> class Impl : public mlir::OpTrait::TraitBase<TypeClass, Impl> {
    inline static mlir::LogicalResult verifyHelper(mlir::Operation *op, int32_t segmentSize) {
      TypeClass c = llvm::cast<TypeClass>(op);
      return affineMapHelpers::verifySizesForMultiAffineOps(
          op, segmentSize, c.getMapOpGroupSizesAttr(), c.getMapOperands(), c.getNumDimsPerMapAttr()
      );
    }

  public:
    static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
      if (TypeClass::template hasTrait<mlir::OpTrait::AttrSizedOperandSegments>()) {
        // If the AttrSizedOperandSegments trait is present, must have `OperandSegmentIndex`.
        static_assert(
            OperandSegmentIndex >= 0,
            "When the `AttrSizedOperandSegments` trait is present, the index of `$mapOperands` "
            "within the `operandSegmentSizes` attribute must be specified."
        );
        mlir::DenseI32ArrayAttr segmentSizes = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
            mlir::OpTrait::AttrSizedOperandSegments<TypeClass>::getOperandSegmentSizeAttr()
        );
        assert(
            OperandSegmentIndex < segmentSizes.size() &&
            "Parameter of `VerifySizesForMultiAffineOps` exceeds the number of ODS-declared "
            "operands"
        );
        return verifyHelper(op, segmentSizes[OperandSegmentIndex]);
      } else {
        // If the trait is not present, the `OperandSegmentIndex` is ignored. Pass `-1` to indicate
        // that the checks against `operandSegmentSizes` should be skipped.
        return verifyHelper(op, -1);
      }
    }
  };
};

template <unsigned N>
inline mlir::ParseResult parseDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand, N> &mapOperands,
    mlir::IntegerAttr &numDims
) {
  return affineMapHelpers::parseDimAndSymbolList(parser, mapOperands, numDims);
}

inline void printDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRange mapOperands,
    mlir::IntegerAttr numDims
) {
  return affineMapHelpers::printDimAndSymbolList(printer, op, mapOperands, numDims);
}

inline mlir::ParseResult parseMultiDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVector<mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>> &multiMapOperands,
    mlir::DenseI32ArrayAttr &numDimsPerMap
) {
  return affineMapHelpers::parseMultiDimAndSymbolList(parser, multiMapOperands, numDimsPerMap);
}

inline void printMultiDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRangeRange multiMapOperands,
    mlir::DenseI32ArrayAttr numDimsPerMap
) {
  return affineMapHelpers::printMultiDimAndSymbolList(printer, op, multiMapOperands, numDimsPerMap);
}

inline mlir::ParseResult parseAttrDictWithWarnings(
    mlir::OpAsmParser &parser, mlir::NamedAttrList &extraAttrs, mlir::OperationState &state
) {
  return affineMapHelpers::parseAttrDictWithWarnings(parser, extraAttrs, state);
}

template <typename ConcreteOp>
inline void printAttrDictWithWarnings(
    mlir::OpAsmPrinter &printer, ConcreteOp op, mlir::DictionaryAttr extraAttrs,
    typename mlir::PropertiesSelector<ConcreteOp>::type state
) {
  return affineMapHelpers::printAttrDictWithWarnings(printer, op, extraAttrs, state);
}

inline mlir::ParseResult parseTemplateParams(mlir::AsmParser &parser, mlir::ArrayAttr &value) {
  mlir::SmallVector<mlir::Attribute> elements;
  auto parseElement = [&]() -> mlir::ParseResult {
    // `?` is a wildcard meaning "infer this parameter"; only valid for tvar-restricted params.
    if (mlir::succeeded(parser.parseOptionalQuestion())) {
      elements.push_back(parser.getBuilder().getIndexAttr(mlir::ShapedType::kDynamic));
      return mlir::success();
    }
    auto attrParseResult = mlir::FieldParser<mlir::Attribute>::parse(parser);
    if (mlir::failed(attrParseResult)) {
      return parser.emitError(
          parser.getCurrentLocation(), "failed to parse template parameter attribute"
      );
    }
    auto emitError = [&parser] {
      return llzk::InFlightDiagnosticWrapper(parser.emitError(parser.getCurrentLocation()));
    };
    mlir::FailureOr<mlir::Attribute> forced = forceIntAttrType(*attrParseResult, emitError);
    if (mlir::failed(forced)) {
      return mlir::failure();
    }
    elements.push_back(*forced);
    return mlir::success();
  };
  auto res = parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::Square, parseElement);
  if (mlir::failed(res)) {
    return res; // parseElement() already emits a sufficient error message
  }
  value = parser.getBuilder().getArrayAttr(elements);
  return mlir::success();
}

// 2 parameter version used by types
inline void printTemplateParams(mlir::AsmPrinter &printer, mlir::ArrayAttr value) {
  printer << '[';
  printAttrs(printer, value.getValue(), ", ");
  printer << ']';
}

// 3 parameter version used by ops
inline void printTemplateParams(mlir::AsmPrinter &printer, void *, mlir::ArrayAttr value) {
  printTemplateParams(printer, value);
}

} // namespace llzk
