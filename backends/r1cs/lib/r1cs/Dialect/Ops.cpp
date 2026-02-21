//===-- Ops.cpp - R1CS operation implementations ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Ops.h"

#include <mlir/IR/OpImplementation.h>

#include <llvm/Support/Casting.h>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "r1cs/Dialect/IR/Ops.cpp.inc"

using namespace mlir;
namespace r1cs {

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
  return parser.parseRegion(*body, args);
}

void CircuitDefOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());

  Block &entry = getBody().front();
  bool hasAttrs = getArgAttrs().has_value();

  if (!entry.empty()) {
    p << " inputs (";
    auto dictAttr = getArgAttrs().value_or(DictionaryAttr::get(getContext()));
    llvm::interleaveComma(entry.getArguments(), p, [&](BlockArgument arg) {
      p << arg << ": ";
      p.printType(arg.getType());

      if (hasAttrs) {
        if (auto attr = dictAttr.get(std::to_string(arg.getArgNumber()))) {
          p << " {";
          p.printAttribute(attr);
          p << '}';
        }
      }
    });
    p << ") ";
  }
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

mlir::Block *CircuitDefOp::addEntryBlock() {
  Region &body = getBody();
  assert(body.empty() && "CircuitOp already has a block");
  Block *block = new Block();
  body.push_back(block);
  return block;
}

LogicalResult CircuitDefOp::verify() {

  // === Step 1: Check that each arg_attrs key is a valid argument index ===
  if (getArgAttrs().has_value()) {
    DictionaryAttr dict = *getArgAttrs();
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

  // === Step 3: Require at least one public signal ===
  if (!foundPublic) {
    return emitOpError() << "at least one signal must be marked public";
  }

  return success();
}

} // namespace r1cs
