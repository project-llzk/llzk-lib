//===-- Ops.cpp - Channel operation implementations ------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Channel/IR/Ops.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/BuiltinTypes.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "llzk/Dialect/Channel/IR/Ops.cpp.inc"

using namespace mlir;
using namespace llzk::felt;

namespace llzk::channel {

namespace {

struct RowOffsetContext {
  Attribute tableOffset;
  SmallVector<int32_t, 1> numDimsPerMap;
  SmallVector<int32_t, 1> mapOpGroupSizes;
  SmallVector<Value, 4> mapOperands;

  bool operator==(const RowOffsetContext &rhs) const {
    return tableOffset == rhs.tableOffset && numDimsPerMap == rhs.numDimsPerMap &&
           mapOperands == rhs.mapOperands;
  }
};

struct RowContext {
  Value root;
  RowOffsetContext offset;

  bool operator==(const RowContext &rhs) const {
    return root == rhs.root && offset == rhs.offset;
  }
};

bool isCurrentRowOffset(Attribute tableOffset) {
  if (!tableOffset) {
    return true;
  }
  auto intAttr = llvm::dyn_cast<IntegerAttr>(tableOffset);
  return intAttr && intAttr.getValue().isZero();
}

RowOffsetContext getRowOffsetContext(component::MemberReadOp op) {
  RowOffsetContext ctx;
  Attribute tableOffset = op.getTableOffset().value_or(nullptr);
  if (isCurrentRowOffset(tableOffset)) {
    return ctx;
  }

  ctx.tableOffset = tableOffset;
  llvm::append_range(ctx.numDimsPerMap, op.getNumDimsPerMap());
  llvm::append_range(ctx.mapOpGroupSizes, op.getMapOpGroupSizes());
  for (OperandRange group : op.getMapOperands()) {
    ctx.mapOperands.append(group.begin(), group.end());
  }
  return ctx;
}

using EmitErrorFnRef = llvm::function_ref<InFlightDiagnostic()>;

LogicalResult mergeRowOffsets(
    RowOffsetContext &lhs, const RowOffsetContext &rhs, EmitErrorFnRef emitError
) {
  if (!rhs.tableOffset) {
    return success();
  }
  if (!lhs.tableOffset) {
    lhs = rhs;
    return success();
  }
  if (lhs == rhs) {
    return success();
  }
  return emitError() << "must be derived from a single row context";
}

FailureOr<RowContext> deriveRowContextFromStorage(Value value, EmitErrorFnRef emitError) {
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
    return RowContext {blockArg, RowOffsetContext {}};
  }

  if (auto readOp = value.getDefiningOp<component::MemberReadOp>()) {
    FailureOr<RowContext> baseCtx = deriveRowContextFromStorage(readOp.getComponent(), emitError);
    if (failed(baseCtx)) {
      return failure();
    }
    RowContext ctx = *baseCtx;
    if (failed(mergeRowOffsets(ctx.offset, getRowOffsetContext(readOp), emitError))) {
      return failure();
    }
    return ctx;
  }

  if (auto readArrayOp = value.getDefiningOp<array::ReadArrayOp>()) {
    return deriveRowContextFromStorage(readArrayOp.getArrRef(), emitError);
  }

  if (Operation *defOp = value.getDefiningOp()) {
    return emitError() << "must be derived from a row-local storage reference, but found '"
                       << defOp->getName() << "'";
  }
  return emitError() << "must be derived from a row-local storage reference";
}

LogicalResult combineRowContexts(
    std::optional<RowContext> lhs, std::optional<RowContext> rhs, std::optional<RowContext> &out,
    EmitErrorFnRef emitError
) {
  if (!lhs) {
    out = rhs;
    return success();
  }
  if (!rhs) {
    out = lhs;
    return success();
  }
  if (*lhs == *rhs) {
    out = lhs;
    return success();
  }
  return emitError() << "must be derived from a single row context";
}

LogicalResult verifyChannelTupleExpr(
    Value value, std::optional<RowContext> &rowCtx, EmitErrorFnRef emitError
) {
  if (!llvm::isa<FeltType>(value.getType())) {
    rowCtx = std::nullopt;
    return success();
  }

  if (llvm::isa<BlockArgument>(value) || value.getDefiningOp<FeltConstantOp>()) {
    rowCtx = std::nullopt;
    return success();
  }

  if (auto addOp = value.getDefiningOp<AddFeltOp>()) {
    std::optional<RowContext> lhsCtx, rhsCtx;
    if (failed(verifyChannelTupleExpr(addOp.getLhs(), lhsCtx, emitError)) ||
        failed(verifyChannelTupleExpr(addOp.getRhs(), rhsCtx, emitError))) {
      return failure();
    }
    return combineRowContexts(lhsCtx, rhsCtx, rowCtx, emitError);
  }

  if (auto subOp = value.getDefiningOp<SubFeltOp>()) {
    std::optional<RowContext> lhsCtx, rhsCtx;
    if (failed(verifyChannelTupleExpr(subOp.getLhs(), lhsCtx, emitError)) ||
        failed(verifyChannelTupleExpr(subOp.getRhs(), rhsCtx, emitError))) {
      return failure();
    }
    return combineRowContexts(lhsCtx, rhsCtx, rowCtx, emitError);
  }

  if (auto mulOp = value.getDefiningOp<MulFeltOp>()) {
    std::optional<RowContext> lhsCtx, rhsCtx;
    if (failed(verifyChannelTupleExpr(mulOp.getLhs(), lhsCtx, emitError)) ||
        failed(verifyChannelTupleExpr(mulOp.getRhs(), rhsCtx, emitError))) {
      return failure();
    }
    return combineRowContexts(lhsCtx, rhsCtx, rowCtx, emitError);
  }

  if (auto negOp = value.getDefiningOp<NegFeltOp>()) {
    return verifyChannelTupleExpr(negOp.getOperand(), rowCtx, emitError);
  }

  if (value.getDefiningOp<component::MemberReadOp>() || value.getDefiningOp<array::ReadArrayOp>()) {
    FailureOr<RowContext> derived = deriveRowContextFromStorage(value, emitError);
    if (failed(derived)) {
      return failure();
    }
    rowCtx = *derived;
    return success();
  }

  return emitError() << "must be a field-native algebraic expression, but found '"
                     << value.getDefiningOp()->getName() << "'";
}

LogicalResult verifyChannelMessageType(Type messageType, EmitErrorFn emitError) {
  auto tupleType = llvm::dyn_cast<TupleType>(messageType);
  if (!tupleType) {
    return emitError() << "message type must be a builtin tuple type but got '" << messageType
                       << "'";
  }
  auto elementTypes = tupleType.getTypes();
  if (elementTypes.empty()) {
    return emitError() << "message type must contain at least one tuple element";
  }
  for (auto [idx, elementType] : llvm::enumerate(elementTypes)) {
    if (!llvm::isa<FeltType>(elementType)) {
      return emitError() << "message type element #" << idx
                         << " must have type '!felt.type' but got '" << elementType << "'";
    }
  }
  return success();
}

TupleType getDeclaredMessageType(ChannelDefOp op) {
  return llvm::cast<TupleType>(op.getMessageType());
}

ParseResult parseTupleOperands(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &tupleOperands
) {
  return parser.parseOperandList(tupleOperands, OpAsmParser::Delimiter::Paren);
}

template <typename ChannelUseOpT>
ParseResult parseChannelUseOpImpl(OpAsmParser &parser, OperationState &result) {
  Builder &builder = parser.getBuilder();
  FlatSymbolRefAttr channelRef;
  SmallVector<OpAsmParser::UnresolvedOperand> tupleOperands;
  SmallVector<Type> tupleTypes;
  OpAsmParser::UnresolvedOperand multOperand;
  Type multType;
  bool hasMult = false;

  if (parser.parseAttribute(
          channelRef, ChannelUseOpT::getChannelRefAttrName(result.name), result.attributes
      ) ||
      parseTupleOperands(parser, tupleOperands)) {
    return failure();
  }

  if (succeeded(parser.parseOptionalLBrace())) {
    hasMult = true;
    if (parser.parseKeyword("mult") || parser.parseEqual() || parser.parseOperand(multOperand) ||
        parser.parseRBrace()) {
      return failure();
    }
  }

  if (tupleOperands.empty()) {
    return parser.emitError(parser.getCurrentLocation())
           << "expected at least one channel tuple operand";
  }

  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon()) {
    return failure();
  }

  tupleTypes.reserve(tupleOperands.size());
  for (size_t i = 0; i < tupleOperands.size(); ++i) {
    Type tupleType;
    if ((i > 0 && parser.parseComma()) || parser.parseType(tupleType)) {
      return failure();
    }
    tupleTypes.push_back(tupleType);
  }
  if (hasMult && (parser.parseComma() || parser.parseType(multType))) {
    return failure();
  }

  if (parser.resolveOperands(
          tupleOperands, tupleTypes, parser.getCurrentLocation(), result.operands
      )) {
    return failure();
  }
  if (hasMult && parser.resolveOperand(multOperand, multType, result.operands)) {
    return failure();
  }

  result.addAttribute(
      "operandSegmentSizes",
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(tupleOperands.size()), hasMult ? 1 : 0})
  );
  return success();
}

template <typename ChannelUseOpT>
void printChannelUseOpImpl(OpAsmPrinter &printer, ChannelUseOpT op) {
  printer << ' ';
  printer.printAttributeWithoutType(op.getChannelRefAttr());
  printer << " (";
  printer.printOperands(op.getTuple());
  printer << ')';
  if (Value mult = op.getMult()) {
    printer << " { mult = " << mult << " }";
  }
  printer.printOptionalAttrDict(op->getAttrs(), {"channel_ref", "operandSegmentSizes"});
  printer << " : ";
  llvm::interleaveComma(op.getTuple(), printer, [&](Value tupleVal) {
    printer.printStrippedAttrOrType(tupleVal.getType());
  });
  if (Value mult = op.getMult()) {
    printer << ", ";
    printer.printStrippedAttrOrType(mult.getType());
  }
}

template <typename ChannelUseOpT>
LogicalResult verifyChannelSymbolUsesImpl(ChannelUseOpT op, SymbolTableCollection &tables) {
  FailureOr<SymbolLookupResult<ChannelDefOp>> target =
      lookupTopLevelSymbol<ChannelDefOp>(tables, op.getChannelRefAttr(), op.getOperation());
  if (failed(target)) {
    return failure();
  }
  ChannelDefOp channelDef = target->get();
  auto declaredMessageType = getDeclaredMessageType(channelDef);
  auto declaredElementTypes = declaredMessageType.getTypes();

  if (op.getTuple().size() != declaredElementTypes.size()) {
    return op.emitOpError() << "expects " << declaredElementTypes.size()
                            << " tuple operands to match channel '" << channelDef.getSymName()
                            << "' but found " << op.getTuple().size();
  }
  for (auto [idx, tupleVal] : llvm::enumerate(op.getTuple())) {
    Type expectedType = declaredElementTypes[idx];
    Type actualType = tupleVal.getType();
    if (actualType != expectedType) {
      return op.emitOpError() << "tuple operand #" << idx << " has wrong type; expected '"
                              << expectedType << "' from channel '" << channelDef.getSymName()
                              << "' but got '" << actualType << "'";
    }
  }

  SmallVector<Type> referencedTypes;
  referencedTypes.reserve(op.getTuple().size() + (op.getMult() ? 1U : 0U));
  for (Value tupleVal : op.getTuple()) {
    referencedTypes.push_back(tupleVal.getType());
  }
  if (Value mult = op.getMult()) {
    referencedTypes.push_back(mult.getType());
  }
  return verifyTypeResolution(tables, op.getOperation(), referencedTypes);
}

template <typename ChannelUseOpT> LogicalResult verifyMultTypeImpl(ChannelUseOpT op) {
  if (Value mult = op.getMult(); mult && !llvm::isa<FeltType>(mult.getType())) {
    return op.emitOpError() << "multiplicity must have type '!felt.type' but got '"
                            << mult.getType() << "'";
  }
  return success();
}

template <typename ChannelUseOpT> LogicalResult verifyTupleExprsImpl(ChannelUseOpT op) {
  std::optional<RowContext> tupleCtx;
  for (auto [idx, tupleVal] : llvm::enumerate(op.getTuple())) {
    auto emitTupleError = [&]() -> InFlightDiagnostic {
      return op.emitOpError() << "tuple operand #" << idx << ' ';
    };

    std::optional<RowContext> exprCtx;
    if (failed(verifyChannelTupleExpr(tupleVal, exprCtx, emitTupleError))) {
      return failure();
    }
    if (failed(combineRowContexts(tupleCtx, exprCtx, tupleCtx, [&]() -> InFlightDiagnostic {
          return op.emitOpError() << "tuple operands ";
        }))) {
      return failure();
    }
  }
  return success();
}

} // namespace

LogicalResult ChannelDefOp::verify() {
  return verifyChannelMessageType(getMessageType(), getEmitOpErrFn(this));
}

ParseResult ChannelPushOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseChannelUseOpImpl<ChannelPushOp>(parser, result);
}

void ChannelPushOp::print(OpAsmPrinter &printer) { printChannelUseOpImpl(printer, *this); }

LogicalResult ChannelPushOp::verify() {
  if (failed(verifyMultTypeImpl(*this))) {
    return failure();
  }
  return verifyTupleExprsImpl(*this);
}

LogicalResult ChannelPushOp::verifySymbolUses(SymbolTableCollection &tables) {
  return verifyChannelSymbolUsesImpl(*this, tables);
}

ParseResult ChannelPullOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseChannelUseOpImpl<ChannelPullOp>(parser, result);
}

void ChannelPullOp::print(OpAsmPrinter &printer) { printChannelUseOpImpl(printer, *this); }

LogicalResult ChannelPullOp::verify() {
  if (failed(verifyMultTypeImpl(*this))) {
    return failure();
  }
  return verifyTupleExprsImpl(*this);
}

LogicalResult ChannelPullOp::verifySymbolUses(SymbolTableCollection &tables) {
  return verifyChannelSymbolUsesImpl(*this, tables);
}

} // namespace llzk::channel
