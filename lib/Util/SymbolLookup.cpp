//===-- SymbolLookup.cpp - LLZK Symbol lookup helpers -----------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementations for symbol lookup helper functions.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Include/IR/Ops.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OwningOpRef.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "llzk-symbol-lookup"

namespace llzk {
using namespace mlir;
using namespace include;

namespace {
SymbolLookupResultUntyped
lookupSymbolRec(SymbolTableCollection &tables, SymbolRefAttr symbol, Operation *symTableOp) {
  // First try a direct lookup via the SymbolTableCollection.  Must use a low-level lookup function
  // in order to properly account for modules that were added due to inlining IncludeOp.
  {
    SmallVector<Operation *, 4> symbolsFound;
    if (succeeded(tables.lookupSymbolIn(symTableOp, symbol, symbolsFound))) {
      SymbolLookupResultUntyped ret(symbolsFound.back());
      for (auto it = symbolsFound.rbegin(); it != symbolsFound.rend(); ++it) {
        Operation *op = *it;
        if (op->hasAttr(LANG_ATTR_NAME)) {
          auto symName = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
          ret.pushNamespace(symName);
          if (!llvm::isa<ModuleOp>(op)) {
            LLVM_DEBUG({ llvm::dbgs() << "[lookupSymbolRec]   tracking op as include\n"; });

            ret.trackIncludeAsName(symName);
          }
        }
      }
      return ret;
    }
  }
  // Otherwise, check if the reference can be found by manually doing a lookup for each part of
  // the reference in turn, traversing through IncludeOp symbols by parsing the included file.
  if (Operation *rootOp = tables.lookupSymbolIn(symTableOp, symbol.getRootReference())) {
    if (IncludeOp rootOpInc = llvm::dyn_cast<IncludeOp>(rootOp)) {
      FailureOr<OwningOpRef<ModuleOp>> otherMod = rootOpInc.openModule();
      if (succeeded(otherMod)) {
        // Create a temporary SymbolTableCollection for caching the external symbols from the
        // included module rather than adding these symbols to the existing SymbolTableCollection
        // because it has no means of removing entries from its internal map and it is not safe to
        // leave the dangling pointers in that map after the external module has been freed.
        SymbolTableCollection external;
        auto result = lookupSymbolRec(external, getTailAsSymbolRefAttr(symbol), otherMod->get());
        if (result) {
          result.manage(std::move(*otherMod), std::move(external));
          auto symName = rootOpInc.getSymName();
          result.pushNamespace(symName);
          result.trackIncludeAsName(symName);
        }
        return result;
      }
    } else if (ModuleOp rootOpMod = llvm::dyn_cast<ModuleOp>(rootOp)) {
      return lookupSymbolRec(tables, getTailAsSymbolRefAttr(symbol), rootOpMod);
    }
  }
  // Otherwise, return empty result
  return SymbolLookupResultUntyped();
}
} // namespace

//===------------------------------------------------------------------===//
// SymbolLookupResultUntyped
//===------------------------------------------------------------------===//

/// Access the internal operation.
Operation *SymbolLookupResultUntyped::operator->() { return op; }
Operation &SymbolLookupResultUntyped::operator*() { return *op; }
Operation &SymbolLookupResultUntyped::operator*() const { return *op; }
Operation *SymbolLookupResultUntyped::get() { return op; }
Operation *SymbolLookupResultUntyped::get() const { return op; }

/// True iff the symbol was found.
SymbolLookupResultUntyped::operator bool() const { return op != nullptr; }

/// Store the resources that the result has to manage the lifetime of.
void SymbolLookupResultUntyped::manage(
    OwningOpRef<ModuleOp> &&ptr, SymbolTableCollection &&tables
) {
  // This may be called multiple times for the same result Operation but we only need to store the
  // resources from the first call because that call will contain the final ModuleOp loaded in a
  // chain of IncludeOp and that is the one which contains the result Operation*.
  if (!managedResources) {
    managedResources = std::make_shared<std::pair<OwningOpRef<ModuleOp>, SymbolTableCollection>>(
        std::make_pair(std::move(ptr), std::move(tables))
    );
  }
}

/// Adds the symbol name from the IncludeOp that caused the module to be loaded.
void SymbolLookupResultUntyped::trackIncludeAsName(llvm::StringRef includeOpSymName) {
  includeSymNameStack.push_back(includeOpSymName);
}

void SymbolLookupResultUntyped::pushNamespace(llvm::StringRef symName) {
  namespaceStack.push_back(symName);
}

void SymbolLookupResultUntyped::prependNamespace(llvm::ArrayRef<llvm::StringRef> ns) {
  std::vector<llvm::StringRef> newNamespace = ns;
  newNamespace.insert(newNamespace.end(), namespaceStack.begin(), namespaceStack.end());
  namespaceStack = newNamespace;
}

//===------------------------------------------------------------------===//
// Within
//===------------------------------------------------------------------===//

Within &Within::operator=(Within &&other) noexcept {
  if (this != &other) {
    from = std::move(other.from);
  }
  return *this;
}

FailureOr<SymbolLookupResultUntyped> Within::lookup(
    SymbolTableCollection &tables, SymbolRefAttr symbol, Operation *origin, bool reportMissing
) && {
  if (SymbolLookupResultUntyped *priorRes = std::get_if<SymbolLookupResultUntyped>(&this->from)) {
    //---- Lookup within an existing result ----//
    // Use the symbol table from prior result if available, otherwise use the parameter.
    SymbolTableCollection *cachedTablesForRes = priorRes->getSymbolTableCache();
    if (!cachedTablesForRes) {
      cachedTablesForRes = &tables;
    }
    if (auto found = lookupSymbolRec(*cachedTablesForRes, symbol, priorRes->op)) {
      assert(!found.managedResources && "should not have loaded additional modules");
      // TODO: not quite sure following is true. If not, the result should contain
      // `priorRes.includeSymNameStack` followed by `found.includeSymNameStack`.
      assert(found.includeSymNameStack.empty() && "should not have loaded additional modules");
      // Move stuff from 'priorRes' to the new result
      found.managedResources = std::move(priorRes->managedResources);
      found.includeSymNameStack = std::move(priorRes->includeSymNameStack);
      return found;
    }
  } else {
    //---- Lookup from a given operation or root (if nullptr) ----//
    Operation *lookupFrom = std::get<Operation *>(this->from);
    if (!lookupFrom) {
      FailureOr<ModuleOp> root = getRootModule(origin);
      if (failed(root)) {
        return failure(); // getRootModule() already emits a sufficient error message
      }
      lookupFrom = root.value();
    }
    if (auto found = lookupSymbolRec(tables, symbol, lookupFrom)) {
      return found;
    }
  }
  // Handle the case where it was not found
  if (reportMissing) {
    return origin->emitOpError() << "references unknown symbol \"" << symbol << '"';
  } else {
    return failure();
  }
}

} // namespace llzk
