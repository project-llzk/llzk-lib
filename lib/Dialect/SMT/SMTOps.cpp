//===- SMTOps.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/SMT/IR/SMTOps.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/TypeSwitch.h>

using namespace mlir;
using namespace llzk::smt;

static bool isValidSetInfoValue(Attribute attr) {
  return TypeSwitch<Attribute, bool>(attr)
      .Case<BoolAttr, IntegerAttr, StringAttr, KeywordAttr, SymbolAttr>([](auto) { return true; })
      .Case<ArrayAttr>([](ArrayAttr arrayAttr) {
    return llvm::all_of(arrayAttr, [](Attribute element) { return isValidSetInfoValue(element); });
  }).Default([](Attribute) { return false; });
}

static void printSetInfoValue(AsmPrinter &printer, Attribute value) {
  TypeSwitch<Attribute>(value)
      .Case<KeywordAttr>([&printer](auto keywordAttr) { printer << keywordAttr.getValue(); })
      .Case<SymbolAttr>([&printer](auto symbolAttr) { printer << symbolAttr.getValue(); })
      .Case<StringAttr, BoolAttr>([&printer](auto attr) { printer.printAttribute(attr); })
      .Case<IntegerAttr>([&printer](auto intAttr) {
    SmallString<32> valueText;
    intAttr.getValue().toStringSigned(valueText);
    printer << valueText;
  }).Case<ArrayAttr>([&printer](ArrayAttr arrayAttr) {
    printer << '(';
    llvm::interleave(arrayAttr, [&printer](Attribute element) {
      printSetInfoValue(printer, element);
    }, [&printer] { printer << ' '; });
    printer << ')';
  });
}

static ParseResult parseSetInfoValue(OpAsmParser &parser, Attribute &value) {
  Builder builder(parser.getContext());

  if (succeeded(parser.parseOptionalLParen())) {
    SmallVector<Attribute> elements;
    while (failed(parser.parseOptionalRParen())) {
      Attribute element;
      if (parseSetInfoValue(parser, element)) {
        return failure();
      }
      elements.push_back(element);
    }
    value = builder.getArrayAttr(elements);
    return success();
  }

  if (succeeded(parser.parseOptionalColon())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    value = KeywordAttr::get(parser.getContext(), (":" + keyword).str());
    return success();
  }

  {
    APInt numeral;
    OptionalParseResult parseResult = parser.parseOptionalInteger(numeral);
    if (!parseResult.has_value()) {
      // no numeral
    } else {
      if (failed(*parseResult)) {
        return failure();
      }
      auto intType = IntegerType::get(parser.getContext(), numeral.getBitWidth());
      value = IntegerAttr::get(intType, numeral);
      return success();
    }
  }

  {
    StringAttr strAttr;
    OptionalParseResult parseResult = parser.parseOptionalAttribute(strAttr, Type());
    if (!parseResult.has_value()) {
      // no string attribute
    } else {
      if (failed(*parseResult)) {
        return failure();
      }
      value = strAttr;
      return success();
    }
  }

  StringRef symbolOrBool;
  if (succeeded(parser.parseOptionalKeyword(&symbolOrBool))) {
    if (symbolOrBool == "true" || symbolOrBool == "false") {
      value = builder.getBoolAttr(symbolOrBool == "true");
    } else {
      value = SymbolAttr::get(parser.getContext(), symbolOrBool);
    }
    return success();
  }

  parser.emitError(parser.getCurrentLocation()) << "expected SMT-LIB set-info value";
  return failure();
}

//===----------------------------------------------------------------------===//
// BVConstantOp
//===----------------------------------------------------------------------===//

LogicalResult BVConstantOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/, ValueRange /*operands*/,
    DictionaryAttr /*attributes*/, OpaqueProperties properties, RegionRange /*regions*/,
    SmallVectorImpl<Type> &inferredReturnTypes
) {
  inferredReturnTypes.push_back(properties.as<Properties *>()->getValue().getType());
  return success();
}

void BVConstantOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 128> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "c" << getValue().getValue() << "_bv" << getValue().getValue().getBitWidth();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult BVConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// DeclareFunOp
//===----------------------------------------------------------------------===//

void DeclareFunOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getNamePrefix().has_value() ? *getNamePrefix() : "");
}

//===----------------------------------------------------------------------===//
// SolverOp
//===----------------------------------------------------------------------===//

LogicalResult SolverOp::verifyRegions() {
  if (getBody()->getTerminator()->getOperands().getTypes() != getResultTypes()) {
    return emitOpError() << "types of yielded values must match return values";
  }
  if (getBody()->getArgumentTypes() != getInputs().getTypes()) {
    return emitOpError() << "block argument types must match the types of the 'inputs'";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// SetInfoOp
//===----------------------------------------------------------------------===//

ParseResult SetInfoOp::parse(OpAsmParser &parser, OperationState &result) {
  SMLoc loc = parser.getCurrentLocation();
  StringAttr keyText;
  Attribute value;

  if (parser.parseAttribute(keyText) || parseSetInfoValue(parser, value) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  auto keyAttr = KeywordAttr::getChecked([&parser, loc]() {
    return parser.emitError(loc);
  }, parser.getContext(), keyText.getValue());
  if (!keyAttr) {
    return failure();
  }

  result.addAttribute("key", keyAttr);
  result.addAttribute("value", value);
  result.location = parser.getEncodedSourceLoc(loc);
  return success();
}

void SetInfoOp::print(OpAsmPrinter &printer) {
  printer << ' ';
  printer.printAttribute(StringAttr::get(getContext(), getKey().getValue()));
  printer << ' ';
  printSetInfoValue(printer, getValueAttr());
  printer.printOptionalAttrDict(getOperation()->getAttrs(), {"key", "value"});
}

LogicalResult SetInfoOp::verify() {
  if (!isValidSetInfoValue(getValueAttr())) {
    return emitOpError(
        "requires an SMT-LIB set-info value built from strings, booleans, "
        "integers, SMT keywords, SMT symbols, or nested lists"
    );
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CheckOp
//===----------------------------------------------------------------------===//

LogicalResult CheckOp::verifyRegions() {
  if (getSatRegion().front().getTerminator()->getOperands().getTypes() != getResultTypes()) {
    return emitOpError() << "types of yielded values in 'sat' region must "
                            "match return values";
  }
  if (getUnknownRegion().front().getTerminator()->getOperands().getTypes() != getResultTypes()) {
    return emitOpError() << "types of yielded values in 'unknown' region must "
                            "match return values";
  }
  if (getUnsatRegion().front().getTerminator()->getOperands().getTypes() != getResultTypes()) {
    return emitOpError() << "types of yielded values in 'unsat' region must "
                            "match return values";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// EqOp
//===----------------------------------------------------------------------===//

static LogicalResult
parseSameOperandTypeVariadicToBoolOp(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputs;
  SMLoc loc = parser.getCurrentLocation();
  Type type;

  if (parser.parseOperandList(inputs) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(type)) {
    return failure();
  }

  result.addTypes(BoolType::get(parser.getContext()));
  if (parser.resolveOperands(
          inputs, SmallVector<Type>(inputs.size(), type), loc, result.operands
      )) {
    return failure();
  }

  return success();
}

ParseResult EqOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSameOperandTypeVariadicToBoolOp(parser, result);
}

void EqOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInputs();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getInputs().front().getType();
}

LogicalResult EqOp::verify() {
  if (getInputs().size() < 2) {
    return emitOpError() << "'inputs' must have at least size 2, but got " << getInputs().size();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// DistinctOp
//===----------------------------------------------------------------------===//

ParseResult DistinctOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseSameOperandTypeVariadicToBoolOp(parser, result);
}

void DistinctOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getInputs();
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getInputs().front().getType();
}

LogicalResult DistinctOp::verify() {
  if (getInputs().size() < 2) {
    return emitOpError() << "'inputs' must have at least size 2, but got " << getInputs().size();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ExtractOp
//===----------------------------------------------------------------------===//

LogicalResult ExtractOp::verify() {
  unsigned rangeWidth = getType().getWidth();
  unsigned inputWidth = cast<BitVectorType>(getInput().getType()).getWidth();
  if (getLowBit() + rangeWidth > inputWidth) {
    return emitOpError(
               "range to be extracted is too big, expected range "
               "starting at index "
           )
           << getLowBit() << " of length " << rangeWidth << " requires input width of at least "
           << (getLowBit() + rangeWidth) << ", but the input width is only " << inputWidth;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ConcatOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> /*location*/, ValueRange operands,
    DictionaryAttr /*attributes*/, OpaqueProperties /*properties*/, RegionRange /*regions*/,
    SmallVectorImpl<Type> &inferredReturnTypes
) {
  inferredReturnTypes.push_back(
      BitVectorType::get(
          context, cast<BitVectorType>(operands[0].getType()).getWidth() +
                       cast<BitVectorType>(operands[1].getType()).getWidth()
      )
  );
  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp
//===----------------------------------------------------------------------===//

LogicalResult RepeatOp::verify() {
  unsigned inputWidth = cast<BitVectorType>(getInput().getType()).getWidth();
  unsigned resultWidth = getType().getWidth();
  if (resultWidth % inputWidth != 0) {
    return emitOpError() << "result bit-vector width must be a multiple of the "
                            "input bit-vector width";
  }

  return success();
}

unsigned RepeatOp::getCount() {
  unsigned inputWidth = cast<BitVectorType>(getInput().getType()).getWidth();
  unsigned resultWidth = getType().getWidth();
  return resultWidth / inputWidth;
}

void RepeatOp::build(OpBuilder &builder, OperationState &state, unsigned count, Value input) {
  int64_t inputWidth = cast<BitVectorType>(input.getType()).getWidth();
  Type resultTy = BitVectorType::get(builder.getContext(), inputWidth * count);
  build(builder, state, resultTy, input);
}

ParseResult RepeatOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type inputType;
  llvm::SMLoc countLoc = parser.getCurrentLocation();

  APInt count;
  if (parser.parseInteger(count) || parser.parseKeyword("times")) {
    return failure();
  }

  if (count.isNonPositive()) {
    return parser.emitError(countLoc) << "integer must be positive";
  }

  llvm::SMLoc inputLoc = parser.getCurrentLocation();
  if (parser.parseOperand(input) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon() || parser.parseType(inputType)) {
    return failure();
  }

  if (parser.resolveOperand(input, inputType, result.operands)) {
    return failure();
  }

  auto bvInputTy = dyn_cast<BitVectorType>(inputType);
  if (!bvInputTy) {
    return parser.emitError(inputLoc) << "input must have bit-vector type";
  }

  // Make sure no assertions can trigger and no silent overflows can happen
  // Bit-width is stored as 'int64_t' parameter in 'BitVectorType'
  const unsigned maxBw = 63;
  if (count.getActiveBits() > maxBw) {
    return parser.emitError(countLoc) << "integer must fit into " << maxBw << " bits";
  }

  // Store multiplication in an APInt twice the size to not have any overflow
  // and check if it can be truncated to 'maxBw' bits without cutting of
  // important bits.
  APInt resultBw = bvInputTy.getWidth() * count.zext(2 * maxBw);
  if (resultBw.getActiveBits() > maxBw) {
    return parser.emitError(countLoc)
           << "result bit-width (provided integer times bit-width of the input "
              "type) must fit into "
           << maxBw << " bits";
  }

  uint64_t val = resultBw.getZExtValue();
  assert(val <= std::numeric_limits<int64_t>::max() && "value too large");
  Type resultTy = BitVectorType::get(parser.getContext(), static_cast<int64_t>(val));
  result.addTypes(resultTy);
  return success();
}

void RepeatOp::print(OpAsmPrinter &printer) {
  printer << ' ' << getCount() << " times " << getInput();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getInput().getType();
}

//===----------------------------------------------------------------------===//
// BoolConstantOp
//===----------------------------------------------------------------------===//

void BoolConstantOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getValue() ? "true" : "false");
}

OpFoldResult BoolConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// IntConstantOp
//===----------------------------------------------------------------------===//

void IntConstantOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) {
  SmallVector<char, 32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << "c" << getValue();
  setNameFn(getResult(), specialName.str());
}

OpFoldResult IntConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValueAttr();
}

void IntConstantOp::print(OpAsmPrinter &p) {
  p << ' ' << getValue();
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/ {"value"});
}

ParseResult IntConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  APInt value;
  if (parser.parseInteger(value)) {
    return failure();
  }

  result.getOrAddProperties<Properties>().setValue(
      IntegerAttr::get(parser.getContext(), APSInt(value))
  );

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  result.addTypes(smt::IntType::get(parser.getContext()));
  return success();
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

template <typename QuantifierOp> static LogicalResult verifyQuantifierRegions(QuantifierOp op) {
  if (op.getBoundVarNames() && op.getBody().getNumArguments() != op.getBoundVarNames()->size()) {
    return op.emitOpError("number of bound variable names must match number of block arguments");
  }
  if (!llvm::all_of(op.getBody().getArgumentTypes(), isAnyNonFuncSMTValueType)) {
    return op.emitOpError() << "bound variables must by any non-function SMT value";
  }

  if (op.getBody().front().getTerminator()->getNumOperands() != 1) {
    return op.emitOpError("must have exactly one yielded value");
  }
  if (!isa<BoolType>(op.getBody().front().getTerminator()->getOperand(0).getType())) {
    return op.emitOpError("yielded value must be of '!smt.bool' type");
  }

  for (auto regionWithIndex : llvm::enumerate(op.getPatterns())) {
    unsigned i = regionWithIndex.index();
    Region &region = regionWithIndex.value();

    if (op.getBody().getArgumentTypes() != region.getArgumentTypes()) {
      return op.emitOpError() << "block argument number and types of the 'body' "
                                 "and 'patterns' region #"
                              << i << " must match";
    }
    if (region.front().getTerminator()->getNumOperands() < 1) {
      return op.emitOpError() << "'patterns' region #" << i
                              << " must have at least one yielded value";
    }

    // All operations in the 'patterns' region must be SMT operations.
    auto result = region.walk([&](Operation *childOp) {
      if (!isa<SMTDialect>(childOp->getDialect())) {
        auto diag = op.emitOpError()
                    << "the 'patterns' region #" << i << " may only contain SMT dialect operations";
        diag.attachNote(childOp->getLoc()) << "first non-SMT operation here";
        return WalkResult::interrupt();
      }

      // There may be no quantifier (or other variable binding) operations in
      // the 'patterns' region.
      if (isa<ForallOp, ExistsOp>(childOp)) {
        auto diag = op.emitOpError() << "the 'patterns' region #" << i
                                     << " must not contain "
                                        "any variable binding operations";
        diag.attachNote(childOp->getLoc()) << "first violating operation here";
        return WalkResult::interrupt();
      }

      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      return failure();
    }
  }

  return success();
}

template <typename Properties>
static void buildQuantifier(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange boundVarTypes,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder,
    std::optional<ArrayRef<StringRef>> boundVarNames,
    function_ref<ValueRange(OpBuilder &, Location, ValueRange)> patternBuilder, uint32_t weight,
    bool noPattern
) {
  odsState.addTypes(BoolType::get(odsBuilder.getContext()));
  if (weight != 0) {
    odsState.getOrAddProperties<Properties>().weight =
        odsBuilder.getIntegerAttr(odsBuilder.getIntegerType(32), weight);
  }
  if (noPattern) {
    odsState.getOrAddProperties<Properties>().noPattern = odsBuilder.getUnitAttr();
  }
  if (boundVarNames.has_value()) {
    SmallVector<Attribute> boundVarNamesList;
    for (StringRef str : *boundVarNames) {
      boundVarNamesList.emplace_back(odsBuilder.getStringAttr(str));
    }
    odsState.getOrAddProperties<Properties>().boundVarNames =
        odsBuilder.getArrayAttr(boundVarNamesList);
  }
  {
    OpBuilder::InsertionGuard guard(odsBuilder);
    Region *region = odsState.addRegion();
    Block *block = odsBuilder.createBlock(region);
    block->addArguments(
        boundVarTypes, SmallVector<Location>(boundVarTypes.size(), odsState.location)
    );
    Value returnVal = bodyBuilder(odsBuilder, odsState.location, block->getArguments());
    odsBuilder.create<llzk::smt::YieldOp>(odsState.location, returnVal);
  }
  if (patternBuilder) {
    Region *region = odsState.addRegion();
    OpBuilder::InsertionGuard guard(odsBuilder);
    Block *block = odsBuilder.createBlock(region);
    block->addArguments(
        boundVarTypes, SmallVector<Location>(boundVarTypes.size(), odsState.location)
    );
    ValueRange returnVals = patternBuilder(odsBuilder, odsState.location, block->getArguments());
    odsBuilder.create<llzk::smt::YieldOp>(odsState.location, returnVals);
  }
}

LogicalResult ForallOp::verify() {
  if (!getPatterns().empty() && getNoPattern()) {
    return emitOpError() << "patterns and the no_pattern attribute must not be "
                            "specified at the same time";
  }

  return success();
}

LogicalResult ForallOp::verifyRegions() { return verifyQuantifierRegions(*this); }

void ForallOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange boundVarTypes,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder,
    std::optional<ArrayRef<StringRef>> boundVarNames,
    function_ref<ValueRange(OpBuilder &, Location, ValueRange)> patternBuilder, uint32_t weight,
    bool noPattern
) {
  buildQuantifier<Properties>(
      odsBuilder, odsState, boundVarTypes, bodyBuilder, boundVarNames, patternBuilder, weight,
      noPattern
  );
}

//===----------------------------------------------------------------------===//
// ExistsOp
//===----------------------------------------------------------------------===//

LogicalResult ExistsOp::verify() {
  if (!getPatterns().empty() && getNoPattern()) {
    return emitOpError() << "patterns and the no_pattern attribute must not be "
                            "specified at the same time";
  }

  return success();
}

LogicalResult ExistsOp::verifyRegions() { return verifyQuantifierRegions(*this); }

void ExistsOp::build(
    OpBuilder &odsBuilder, OperationState &odsState, TypeRange boundVarTypes,
    function_ref<Value(OpBuilder &, Location, ValueRange)> bodyBuilder,
    std::optional<ArrayRef<StringRef>> boundVarNames,
    function_ref<ValueRange(OpBuilder &, Location, ValueRange)> patternBuilder, uint32_t weight,
    bool noPattern
) {
  buildQuantifier<Properties>(
      odsBuilder, odsState, boundVarTypes, bodyBuilder, boundVarNames, patternBuilder, weight,
      noPattern
  );
}

#define GET_OP_CLASSES
#include "llzk/Dialect/SMT/IR/SMT.cpp.inc"
