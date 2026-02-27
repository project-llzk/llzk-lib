//===-- AffineHelper.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Util/AffineHelper.h"
#include "llzk/Util/Compare.h"

#include <numeric>

using namespace mlir;

namespace llzk::affineMapHelpers {

namespace {

ParseResult parseDimAndSymbolListImpl(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mapOperands,
    int32_t &numDims
) {
  // Parse the required dimension operands.
  if (parser.parseOperandList(mapOperands, OpAsmParser::Delimiter::Paren)) {
    return failure();
  }
  // Store number of dimensions for validation by caller.
  numDims = llzk::checkedCast<int32_t>(mapOperands.size());

  // Parse the optional symbol operands.
  return parser.parseOperandList(mapOperands, OpAsmParser::Delimiter::OptionalSquare);
}

void printDimAndSymbolListImpl(OpAsmPrinter &printer, OperandRange mapOperands, size_t numDims) {
  printer << '(' << mapOperands.take_front(numDims) << ')';
  if (mapOperands.size() > numDims) {
    printer << '[' << mapOperands.drop_front(numDims) << ']';
  }
}
} // namespace

ParseResult parseDimAndSymbolList(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &mapOperands,
    IntegerAttr &numDims
) {
  int32_t numDimsRes = -1;
  ParseResult res = parseDimAndSymbolListImpl(parser, mapOperands, numDimsRes);
  numDims = parser.getBuilder().getIndexAttr(numDimsRes);
  return res;
}

void printDimAndSymbolList(
    OpAsmPrinter &printer, Operation *, OperandRange mapOperands, IntegerAttr numDims
) {
  printDimAndSymbolListImpl(printer, mapOperands, numDims.getInt());
}

ParseResult parseMultiDimAndSymbolList(
    OpAsmParser &parser,
    SmallVectorImpl<SmallVector<OpAsmParser::UnresolvedOperand>> &multiMapOperands,
    DenseI32ArrayAttr &numDimsPerMap
) {
  SmallVector<int32_t> numDimsPerMapRes;
  auto parseEach = [&]() -> ParseResult {
    SmallVector<OpAsmParser::UnresolvedOperand> nextMapOps;
    int32_t nextMapDims = -1;
    ParseResult res = parseDimAndSymbolListImpl(parser, nextMapOps, nextMapDims);
    numDimsPerMapRes.push_back(nextMapDims);
    multiMapOperands.push_back(nextMapOps);
    return res;
  };
  ParseResult res = parser.parseCommaSeparatedList(AsmParser::Delimiter::None, parseEach);

  numDimsPerMap = parser.getBuilder().getDenseI32ArrayAttr(numDimsPerMapRes);
  return res;
}

void printMultiDimAndSymbolList(
    OpAsmPrinter &printer, Operation *, OperandRangeRange multiMapOperands,
    DenseI32ArrayAttr numDimsPerMap
) {
  size_t count = numDimsPerMap.size();
  assert(multiMapOperands.size() == count);
  llvm::interleaveComma(llvm::seq<size_t>(0, count), printer.getStream(), [&](size_t i) {
    printDimAndSymbolListImpl(printer, multiMapOperands[i], numDimsPerMap[i]);
  });
}

ParseResult
parseAttrDictWithWarnings(OpAsmParser &parser, NamedAttrList &extraAttrs, OperationState &state) {
  // Replicate what ODS generates w/o the custom<AttrDictWithWarnings> directive
  llvm::SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(extraAttrs)) {
    return failure();
  }
  if (failed(state.name.verifyInherentAttrs(extraAttrs, [&]() {
    return parser.emitError(loc) << '\'' << state.name.getStringRef() << "' op ";
  }))) {
    return failure();
  }
  // Ignore, with warnings, any attributes that are specified and shouldn't be
  for (StringAttr skipName : state.name.getAttributeNames()) {
    if (extraAttrs.erase(skipName)) {
      auto msg = "Ignoring attribute '" + skipName.getValue() +
                 "' because it must be computed automatically.";
      mlir::emitWarning(parser.getEncodedSourceLoc(loc), msg).report();
    }
  }
  // There is no failure from this last check, only warnings
  return success();
}

namespace {
inline InFlightDiagnostic msgInstantiationGroupAttrMismatch(
    Operation *op, size_t mapOpGroupSizesCount, size_t mapOperandsSize
) {
  return op->emitOpError().append(
      "map instantiation group count (", mapOperandsSize,
      ") does not match with length of 'mapOpGroupSizes' attribute (", mapOpGroupSizesCount, ")"
  );
}
} // namespace

LogicalResult verifySizesForMultiAffineOps(
    Operation *op, int32_t segmentSize, ArrayRef<int32_t> mapOpGroupSizes,
    OperandRangeRange mapOperands, ArrayRef<int32_t> numDimsPerMap
) {
  // Ensure the `mapOpGroupSizes` and `operandSegmentSizes` attributes agree.
  // NOTE: the ODS generates verifyValueSizeAttr() which ensures 'mapOpGroupSizes' has no negative
  // elements and its sum is equal to the operand group size (which is similar to this check).
  // If segmentSize < 0 the check is validated regardless of the difference.
  int32_t totalMapOpGroupSizes = std::reduce(mapOpGroupSizes.begin(), mapOpGroupSizes.end());
  if (totalMapOpGroupSizes != segmentSize && segmentSize >= 0) {
    // Since `mapOpGroupSizes` and `segmentSize` are computed this should never happen.
    return op->emitOpError().append(
        "number of operands for affine map instantiation (", totalMapOpGroupSizes,
        ") does not match with the total size (", segmentSize,
        ") specified in attribute 'operandSegmentSizes'"
    );
  }

  // Ensure the size of `mapOperands` and its two list attributes are the same.
  // This will be true if the op was constructed via parseMultiDimAndSymbolList()
  //  but when constructed via the build() API, it can be inconsistent.
  size_t count = mapOpGroupSizes.size();
  if (mapOperands.size() != count) {
    return msgInstantiationGroupAttrMismatch(op, count, mapOperands.size());
  }
  if (numDimsPerMap.size() != count) {
    // Tested in CallOpTests.cpp
    return op->emitOpError().append(
        "length of 'numDimsPerMap' attribute (", numDimsPerMap.size(),
        ") does not match with length of 'mapOpGroupSizes' attribute (", count, ")"
    );
  }

  // Verify the following:
  //   1. 'mapOperands' element sizes match 'mapOpGroupSizes' values
  //   2. each 'numDimsPerMap' is <= corresponding 'mapOpGroupSizes'
  LogicalResult aggregateResult = success();
  for (size_t i = 0; i < count; ++i) {
    auto currMapOpGroupSize = mapOpGroupSizes[i];
    if (std::cmp_not_equal(mapOperands[i].size(), currMapOpGroupSize)) {
      // Since `mapOpGroupSizes` is computed this should never happen.
      aggregateResult = op->emitOpError().append(
          "map instantiation group ", i, " operand count (", mapOperands[i].size(),
          ") does not match group ", i, " size in 'mapOpGroupSizes' attribute (",
          currMapOpGroupSize, ")"
      );
    } else if (std::cmp_greater(numDimsPerMap[i], currMapOpGroupSize)) {
      // Tested in CallOpTests.cpp
      aggregateResult = op->emitOpError().append(
          "map instantiation group ", i, " dimension count (", numDimsPerMap[i], ") exceeds group ",
          i, " size in 'mapOpGroupSizes' attribute (", currMapOpGroupSize, ")"
      );
    }
  }
  return aggregateResult;
}

LogicalResult verifyAffineMapInstantiations(
    OperandRangeRange mapOps, ArrayRef<int32_t> numDimsPerMap, ArrayRef<AffineMapAttr> mapAttrs,
    Operation *origin
) {
  size_t count = numDimsPerMap.size();
  if (mapOps.size() != count) {
    return msgInstantiationGroupAttrMismatch(origin, count, mapOps.size());
  }

  // Ensure there is one OperandRange for each AffineMapAttr
  if (mapAttrs.size() != count) {
    // Tested in array_build_fail.llzk, call_with_affinemap_fail.llzk, CallOpTests.cpp, and
    // CreateArrayOpTests.cpp
    return origin->emitOpError().append(
        "map instantiation group count (", count,
        ") does not match the number of affine map instantiations (", mapAttrs.size(),
        ") required by the type"
    );
  }

  // Ensure the affine map identifier counts match the instantiation.
  // Rather than immediately returning on failure, we check all dimensions and aggregate to provide
  // as many errors are possible in a single verifier run.
  LogicalResult aggregateResult = success();
  for (size_t i = 0; i < count; ++i) {
    AffineMap map = mapAttrs[i].getAffineMap();
    if (std::cmp_not_equal(map.getNumDims(), numDimsPerMap[i])) {
      // Tested in array_build_fail.llzk and call_with_affinemap_fail.llzk
      aggregateResult = origin->emitOpError().append(
          "instantiation of map ", i, " expected ", map.getNumDims(), " but found ",
          numDimsPerMap[i], " dimension values in ()"
      );
    } else if (std::cmp_not_equal(map.getNumInputs(), mapOps[i].size())) {
      // Tested in array_build_fail.llzk and call_with_affinemap_fail.llzk
      aggregateResult = origin->emitOpError().append(
          "instantiation of map ", i, " expected ", map.getNumSymbols(), " but found ",
          (mapOps[i].size() - numDimsPerMap[i]), " symbol values in []"
      );
    }
  }
  return aggregateResult;
}

} // namespace llzk::affineMapHelpers
