//===-- Ops.cpp - R1CS operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Ops.h"

#include "r1cs/Dialect/IR/Constants.h"

#include <mlir/IR/OpImplementation.h>

#include <llvm/Support/Casting.h>

#include <limits>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "r1cs/Dialect/IR/Ops.cpp.inc"

using namespace mlir;

namespace r1cs {

void CircuitDefOp::build(OpBuilder &builder, OperationState &state, StringRef name) {
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addRegion();
}

ParseResult CircuitDefOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse symbol name
  StringAttr symName;
  if (parser.parseSymbolName(symName, SymbolTable::getSymbolAttrName(), result.attributes)) {
    return failure();
  }

  SmallVector<OpAsmParser::Argument> args;
  DenseMap<unsigned, Attribute> perArgAttrs;

  if (succeeded(parser.parseOptionalKeyword("inputs"))) {
    if (parser.parseLParen()) {
      return failure();
    }

    unsigned idx = 0;
    do {
      OpAsmParser::Argument arg;
      Type type;
      if (parser.parseArgument(arg) || parser.parseColonType(type)) {
        return failure();
      }
      arg.type = type;

      // Try to parse an optional `{...}` attr
      NamedAttrList attrList;
      if (succeeded(parser.parseOptionalAttrDict(attrList))) {
        if (auto pubAttr = attrList.get(PublicAttr::getMnemonic())) {
          perArgAttrs[idx] = pubAttr;
        }
      }

      args.push_back(arg);
      idx++;
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen()) {
      return failure();
    }
  }

  if (!perArgAttrs.empty()) {
    NamedAttrList attrs;
    for (auto [i, attr] : perArgAttrs) {
      attrs.append(std::to_string(i), attr);
    }
    result.addAttribute("arg_attrs", DictionaryAttr::get(parser.getContext(), attrs));
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, args)) {
    return failure();
  }
  return parser.parseOptionalAttrDict(result.attributes);
}

void CircuitDefOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());

  Block &entry = getBody().front();
  bool hasAttrs = getArgAttrs().has_value();

  if (entry.getNumArguments() != 0) {
    p << " inputs (";
    auto dictAttr = getArgAttrs().value_or(DictionaryAttr::get(getContext()));
    llvm::interleaveComma(entry.getArguments(), p, [&](BlockArgument arg) {
      p << arg << ": ";
      p.printType(arg.getType());

      if (hasAttrs) {
        if (auto attr = dictAttr.get(std::to_string(arg.getArgNumber()))) {
          if (auto pubAttr = dyn_cast<PublicAttr>(attr)) {
            p << " {" << PublicAttr::getMnemonic() << " = ";
            p.printAttribute(pubAttr);
            p << '}';
          }
        }
      }
    });
    p << ')';
  }
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), {"sym_name", "arg_attrs"});
}

mlir::Block *CircuitDefOp::addEntryBlock() {
  Region &body = getBody();
  assert(body.empty() && "CircuitOp already has a block");
  Block *block = new Block();
  body.push_back(block);
  return block;
}

namespace {

/// Check the format ordering when the circuit supplies explicit input wire IDs.
LogicalResult verifyExplicitWireIds(CircuitDefOp circuit) {
  auto inputs = circuit->getAttrOfType<DenseI64ArrayAttr>(INPUT_LABELS_ATTR);
  if (!inputs) {
    return success();
  }
  DenseMap<WireKind, SmallVector<int64_t>> idsByKind;
  for (auto signal : circuit.getBody().front().getOps<SignalDefOp>()) {
    WireKind kind = signal.getPub() ? WireKind::PublicOutput : WireKind::Internal;
    idsByKind[kind].push_back(signal.getLabel());
  }
  auto argAttrs = circuit.getArgAttrs().value_or(DictionaryAttr::get(circuit.getContext()));
  for (auto [index, id] : llvm::enumerate(inputs.asArrayRef())) {
    bool isPublic = isa_and_nonnull<PublicAttr>(argAttrs.get(std::to_string(index)));
    WireKind kind = isPublic ? WireKind::PublicInput : WireKind::PrivateInput;
    idsByKind[kind].push_back(id);
  }
  int64_t expected = FIRST_SIGNAL_WIRE_ID;
  for (WireKind kind : WIRE_KIND_ORDER) {
    for (int64_t id : idsByKind[kind]) {
      if (id != expected++) {
        return circuit.emitError() << '\'' << INPUT_LABELS_ATTR
                                   << "' must be ordered as public outputs, public inputs, private "
                                      "inputs, then internal wires";
      }
    }
  }
  return success();
}

} // namespace

LogicalResult CircuitDefOp::verify() {

  // === Step 1: Check that each arg_attrs key is a valid argument index ===
  auto argAttrsOpt = getArgAttrs();
  if (argAttrsOpt.has_value()) {
    DictionaryAttr dict = *argAttrsOpt;
    unsigned numArgs = getBody().front().getNumArguments();

    for (auto attr : dict) {
      unsigned index = 0;
      if (attr.getName().strref().getAsInteger(10, index)) {
        return emitOpError() << "invalid key '" << attr.getName()
                             << "' in 'arg_attrs': expected integer index";
      }
      if (index >= numArgs) {
        return emitOpError() << "argument index " << index << " out of bounds (only " << numArgs
                             << " arguments)";
      }
      if (!llvm::isa<r1cs::PublicAttr>(attr.getValue())) {
        return emitOpError() << "invalid attribute for argument " << index << ": expected "
                             << PublicAttr::name;
      }
    }
  }

  // === Step 2: Check that signal labels are unique ===
  DenseSet<uint32_t> seenLabels;
  if (auto labels = (*this)->getAttr(INPUT_LABELS_ATTR)) {
    auto values = dyn_cast<DenseI64ArrayAttr>(labels);
    if (!values || values.size() != getBody().front().getNumArguments()) {
      return emitError() << '\'' << INPUT_LABELS_ATTR << "' must contain one i64 label per input";
    }
    for (int64_t label : values.asArrayRef()) {
      if (label < FIRST_SIGNAL_WIRE_ID || label > std::numeric_limits<uint32_t>::max()) {
        return emitError() << '\'' << INPUT_LABELS_ATTR << "' must be in [" << FIRST_SIGNAL_WIRE_ID
                           << ", " << std::numeric_limits<uint32_t>::max() << ']';
      }
      if (!seenLabels.insert(label).second) {
        return emitError() << '\'' << INPUT_LABELS_ATTR << "' must not contain duplicate labels";
      }
    }
  }
  bool foundPublic = false;

  for (auto &op : getBody().front()) {
    if (auto def = dyn_cast<SignalDefOp>(op)) {
      uint32_t label = def.getLabel();
      if (!seenLabels.insert(label).second) {
        return def.emitOpError() << "duplicate signal label: " << label;
      }

      if (def.getPub().has_value()) {
        foundPublic = true;
      }
    }
  }

  if (failed(verifyExplicitWireIds(*this))) {
    return failure();
  }

  // === Step 3: Require at least one public signal ===
  if (!foundPublic) {
    return emitOpError("at least one signal must be marked public");
  }

  return success();
}

} // namespace r1cs
