//===-- AffineHelper.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/Compare.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/ValueRange.h>

#include <llvm/ADT/SmallVector.h>

/// Group together all implementation related to AffineMap type parameters.
namespace llzk::affineMapHelpers {

/// Parses dimension and symbol list for an AffineMap instantiation.
mlir::ParseResult parseDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &mapOperands,
    mlir::IntegerAttr &numDims
);

/// Prints dimension and symbol list for an AffineMap instantiation.
void printDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRange mapOperands,
    mlir::IntegerAttr numDims
);

/// Parses comma-separated list of multiple AffineMap instantiations.
mlir::ParseResult parseMultiDimAndSymbolList(
    mlir::OpAsmParser &parser,
    mlir::SmallVectorImpl<mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand>>
        &multiMapOperands,
    mlir::DenseI32ArrayAttr &numDimsPerMap
);

/// Prints comma-separated list of multiple AffineMap instantiations.
void printMultiDimAndSymbolList(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRangeRange multiMapOperands,
    mlir::DenseI32ArrayAttr numDimsPerMap
);

/// This custom parse/print AttrDictWithWarnings is necessary to directly check what 'attr-dict' is
/// parsed from the input. Waiting until the `verify()` function will not work because the generated
/// `parse()` function automatically computes and initializes the attributes.
mlir::ParseResult parseAttrDictWithWarnings(
    mlir::OpAsmParser &parser, mlir::NamedAttrList &extraAttrs, mlir::OperationState &state
);

template <typename ConcreteOp>
inline void printAttrDictWithWarnings(
    mlir::OpAsmPrinter &printer, ConcreteOp /*op*/, mlir::DictionaryAttr extraAttrs,
    typename ConcreteOp::Properties /*state*/
) {
  printer.printOptionalAttrDict(extraAttrs.getValue(), ConcreteOp::getAttributeNames());
}

/// Implements the ODS trait with the same name. Produces errors if there is an inconsistency in the
/// various attributes/values that are used to support affine map instantiation in the op.
mlir::LogicalResult verifySizesForMultiAffineOps(
    mlir::Operation *op, int32_t segmentSize, mlir::ArrayRef<int32_t> mapOpGroupSizes,
    mlir::OperandRangeRange mapOperands, mlir::ArrayRef<int32_t> numDimsPerMap
);

/// Produces errors if there is an inconsistency between the attributes/values that are used to
/// support affine map instantiation in the op and the AffineMapAttr list collected from the type.
mlir::LogicalResult verifyAffineMapInstantiations(
    mlir::OperandRangeRange mapOps, mlir::ArrayRef<int32_t> numDimsPerMap,
    mlir::ArrayRef<mlir::AffineMapAttr> mapAttrs, mlir::Operation *origin
);

/// Utility for build() functions that initializes the `operandSegmentSizes`, `mapOpGroupSizes`, and
/// `numDimsPerMap` attributes for an Op that performs affine map instantiations.
///
/// Note: This function supports Ops with 2 ODS-defined operand segments with the second being the
/// size of the `mapOperands` segment and the first provided by the `firstSegmentSize` parameter.
template <typename OpClass>
inline typename OpClass::Properties &buildInstantiationAttrs(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
    mlir::ArrayRef<mlir::ValueRange> mapOperands, mlir::DenseI32ArrayAttr numDimsPerMap,
    int32_t firstSegmentSize = 0
) {
  int32_t mapOpsSegmentSize = 0;
  mlir::SmallVector<int32_t> rangeSegments;
  for (mlir::ValueRange r : mapOperands) {
    odsState.addOperands(r);
    int32_t s = llzk::checkedCast<int32_t>(r.size());
    rangeSegments.push_back(s);
    mapOpsSegmentSize += s;
  }
  typename OpClass::Properties &props = odsState.getOrAddProperties<typename OpClass::Properties>();
  props.setMapOpGroupSizes(odsBuilder.getDenseI32ArrayAttr(rangeSegments));
  props.setOperandSegmentSizes({firstSegmentSize, mapOpsSegmentSize});
  if (numDimsPerMap) {
    props.setNumDimsPerMap(numDimsPerMap);
  }
  return props;
}

/// Utility for build() functions that initializes the `mapOpGroupSizes`, and
/// `numDimsPerMap` attributes for an Op that performs affine map instantiations in the case were
/// the op does not have two variadic sets of operands.
template <typename OpClass>
inline void buildInstantiationAttrsNoSegments(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState,
    mlir::ArrayRef<mlir::ValueRange> mapOperands, mlir::DenseI32ArrayAttr numDimsPerMap
) {
  mlir::SmallVector<int32_t> rangeSegments;
  for (mlir::ValueRange r : mapOperands) {
    odsState.addOperands(r);
    int32_t s = llzk::checkedCast<int32_t>(r.size());
    rangeSegments.push_back(s);
  }
  typename OpClass::Properties &props = odsState.getOrAddProperties<typename OpClass::Properties>();
  props.setMapOpGroupSizes(odsBuilder.getDenseI32ArrayAttr(rangeSegments));
  if (numDimsPerMap) {
    props.setNumDimsPerMap(numDimsPerMap);
  }
}

/// Utility for build() functions that initializes the `operandSegmentSizes`, `mapOpGroupSizes`, and
/// `numDimsPerMap` attributes for an Op that supports affine map instantiations but in the case
/// where there are none.
template <typename OpClass>
inline typename OpClass::Properties &buildInstantiationAttrsEmpty(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, int32_t firstSegmentSize = 0
) {
  typename OpClass::Properties &props = odsState.getOrAddProperties<typename OpClass::Properties>();
  // `operandSegmentSizes` = [ firstSegmentSize, mapOperands.size ]
  props.setOperandSegmentSizes({firstSegmentSize, 0});
  // There are no affine map operands so initialize the related properties as empty arrays.
  props.setMapOpGroupSizes(odsBuilder.getDenseI32ArrayAttr({}));
  props.setNumDimsPerMap(odsBuilder.getDenseI32ArrayAttr({}));
  return props;
}

/// Utility for build() functions that initializes the `mapOpGroupSizes`, and
/// `numDimsPerMap` attributes for an Op that supports affine map instantiations but in the case
/// where there are none.
template <typename OpClass>
inline typename OpClass::Properties &buildInstantiationAttrsEmptyNoSegments(
    mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState
) {
  typename OpClass::Properties &props = odsState.getOrAddProperties<typename OpClass::Properties>();
  // There are no affine map operands so initialize the related properties as empty arrays.
  props.setMapOpGroupSizes(odsBuilder.getDenseI32ArrayAttr({}));
  props.setNumDimsPerMap(odsBuilder.getDenseI32ArrayAttr({}));
  return props;
}

} // namespace llzk::affineMapHelpers
