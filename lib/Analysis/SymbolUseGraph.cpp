//===-- SymbolUseGraph.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SymbolUseGraph.h"
#include "llzk/Dialect/Shared/OpHelpers.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/GraphWriter.h>

using namespace mlir;

namespace llzk {

//===----------------------------------------------------------------------===//
// SymbolUseGraphNode
//===----------------------------------------------------------------------===//

void SymbolUseGraphNode::addSuccessor(SymbolUseGraphNode *node) {
  if (this->successors.insert(node)) {
    node->predecessors.insert(this);
  }
}

void SymbolUseGraphNode::removeSuccessor(SymbolUseGraphNode *node) {
  if (this->successors.remove(node)) {
    node->predecessors.remove(this);
  }
}

FailureOr<SymbolLookupResultUntyped>
SymbolUseGraphNode::lookupSymbol(SymbolTableCollection &tables, bool reportMissing) const {
  if (!isRealNode()) {
    return failure();
  }
  Operation *lookupFrom = getSymbolPathRoot().getOperation();
  auto res = lookupSymbolIn(tables, getSymbolPath(), lookupFrom, lookupFrom, reportMissing);
  if (succeeded(res) || !reportMissing) {
    return res;
  }
  // This is likely an error in the use graph and not a case that should ever happen.
  return lookupFrom->emitError().append(
      "Could not find symbol referenced in UseGraph: ", getSymbolPath()
  );
}

//===----------------------------------------------------------------------===//
// SymbolUseGraph
//===----------------------------------------------------------------------===//

namespace {

template <typename R>
R getPathAndCall(SymbolOpInterface defOp, llvm::function_ref<R(ModuleOp, SymbolRefAttr)> callback) {
  assert(defOp); // pre-condition

  ModuleOp foundRoot;
  FailureOr<SymbolRefAttr> path = llzk::getPathFromRoot(defOp, &foundRoot);
  if (failed(path)) {
    // This occurs if there is no root module with LANG_ATTR_NAME attribute
    // or there is an unnamed module between the root module and the symbol.
    auto diag = defOp.emitError("in SymbolUseGraph, failed to build symbol path");
    diag.attachNote(defOp.getLoc()).append("for this SymbolOp");
    diag.report();
    return nullptr;
  }
  return callback(foundRoot, path.value());
}

} // namespace

SymbolUseGraph::SymbolUseGraph(SymbolOpInterface rootSymbolOp) {
  assert(rootSymbolOp->hasTrait<OpTrait::SymbolTable>());
  buildGraph(rootSymbolOp);
}

/// Get (add if not present) the graph node for the "user" symbol def op.
SymbolUseGraphNode *SymbolUseGraph::getSymbolUserNode(const SymbolTable::SymbolUse &u) {
  SymbolOpInterface userSymbol = getSelfOrParentOfType<SymbolOpInterface>(u.getUser());
  return getPathAndCall<SymbolUseGraphNode *>(
      userSymbol, [this, &userSymbol](ModuleOp r, SymbolRefAttr p) {
    auto n = this->getOrAddNode(r, p, nullptr);
    n->opsThatUseTheSymbol.insert(userSymbol);
    return n;
  }
  );
}

void SymbolUseGraph::buildGraph(SymbolOpInterface symbolOp) {
  auto walkFn = [this](Operation *op, bool) {
    assert(op->hasTrait<OpTrait::SymbolTable>());
    FailureOr<ModuleOp> opRootModule = llzk::getRootModule(op);
    if (failed(opRootModule)) {
      return;
    }

    SymbolTableCollection tables;
    if (auto usesOpt = llzk::getSymbolUses(&op->getRegion(0))) {
      // Create child node for each Symbol use, as successor of the user Symbol op.
      for (SymbolTable::SymbolUse u : usesOpt.value()) {
        bool isStructParam = false;
        Operation *user = u.getUser();
        SymbolRefAttr symRef = u.getSymbolRef();
        // Pending [LLZK-272] only a heuristic approach is possible. Check for FlatSymbolRefAttr
        // where the user is a MemberRefOpInterface or the user is located within a StructDefOp and
        // append the StructDefOp path with the FlatSymbolRefAttr.
        if (FlatSymbolRefAttr flatSymRef = llvm::dyn_cast<FlatSymbolRefAttr>(symRef)) {
          if (auto fref = llvm::dyn_cast<component::MemberRefOpInterface>(user);
              fref && fref.getMemberNameAttr() == flatSymRef) {
            symRef = llzk::appendLeaf(fref.getStructType().getNameRef(), flatSymRef);
          } else if (auto userStruct = getSelfOrParentOfType<component::StructDefOp>(user)) {
            StringAttr localName = flatSymRef.getAttr();
            isStructParam = userStruct.hasParamNamed(localName);
            if (isStructParam || tables.getSymbolTable(userStruct).lookup(localName)) {
              // If 'flatSymRef' is defined in the SymbolTable for 'userStruct' then it's
              // a local symbol so prepend the full path of the struct itself.
              auto parentPath = llzk::getPathFromRoot(userStruct);
              assert(succeeded(parentPath));
              symRef = llzk::appendLeaf(parentPath.value(), flatSymRef);
            }
          }
        }
        auto node = this->getOrAddNode(opRootModule.value(), symRef, getSymbolUserNode(u));
        node->isStructConstParam = isStructParam;
        node->opsThatUseTheSymbol.insert(user);
      }
    }
  };
  SymbolTable::walkSymbolTables(symbolOp.getOperation(), true, walkFn);

  // Find all nodes with no successors and add the tail node as successor.
  for (SymbolUseGraphNode *n : nodesIter()) {
    if (!n->hasSuccessor()) {
      n->addSuccessor(&tail);
    }
  }
}

SymbolUseGraphNode *SymbolUseGraph::getOrAddNode(
    ModuleOp pathRoot, SymbolRefAttr path, SymbolUseGraphNode *predecessorNode
) {
  NodeMapKeyT key = std::make_pair(pathRoot, path);
  std::unique_ptr<SymbolUseGraphNode> &nodeRef = nodes[key];
  if (!nodeRef) {
    nodeRef.reset(new SymbolUseGraphNode(pathRoot, path));
    // When creating a new node, ensure it's attached to the graph, either as successor
    // to the predecessor node (if given) else as successor to the root node.
    if (predecessorNode) {
      predecessorNode->addSuccessor(nodeRef.get());
    } else {
      root.addSuccessor(nodeRef.get());
    }
  } else if (predecessorNode) {
    // When the node already exists and an additional predecessor node is given, add the node as a
    // successor to the given predecessor node and detach from the 'root' (unless it's a self edge).
    SymbolUseGraphNode *node = nodeRef.get();
    predecessorNode->addSuccessor(node);
    if (node != predecessorNode) {
      root.removeSuccessor(node);
    }
  }
  return nodeRef.get();
}

const SymbolUseGraphNode *SymbolUseGraph::lookupNode(ModuleOp pathRoot, SymbolRefAttr path) const {
  NodeMapKeyT key = std::make_pair(pathRoot, path);
  const auto *it = nodes.find(key);
  return it == nodes.end() ? nullptr : it->second.get();
}

const SymbolUseGraphNode *SymbolUseGraph::lookupNode(SymbolOpInterface symbolDef) const {
  return getPathAndCall<const SymbolUseGraphNode *>(symbolDef, [this](ModuleOp r, SymbolRefAttr p) {
    return this->lookupNode(r, p);
  });
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

std::string SymbolUseGraphNode::toString(bool showLocations) const {
  return buildStringViaPrint(*this, showLocations);
}

namespace {

inline void safeAppendPathRoot(llvm::raw_ostream &os, ModuleOp root) {
  if (root) {
    FailureOr<SymbolRefAttr> unambiguousRoot = getPathFromTopRoot(root);
    if (succeeded(unambiguousRoot)) {
      os << unambiguousRoot.value() << '\n';
    } else {
      os << "<<unknown path>>\n";
    }
  } else {
    os << "<<NULL MODULE>>\n";
  }
}

} // namespace

void SymbolUseGraphNode::print(
    llvm::raw_ostream &os, bool showLocations, std::string locationLinePrefix
) const {
  os << '\'' << symbolPath << '\'';
  if (isStructConstParam) {
    os << " (struct param)";
  }
  os << " with root module ";
  safeAppendPathRoot(os, symbolPathRoot);
  if (showLocations) {
    // Print the user op locations (sorted for stable output). Printing only the location rather
    // than the full Operation gives a short (single-line) format that's still useful for human
    // debugging.
    llvm::SmallSet<mlir::Location, 3, LocationComparator> locations;
    for (Operation *user : getUserOps()) {
      locations.insert(user->getLoc());
    }
    for (Location loc : locations) {
      os << locationLinePrefix << loc << '\n';
    }
  }
}

void SymbolUseGraph::print(llvm::raw_ostream &os) const {
  const SymbolUseGraphNode *rootPtr = &this->root;

  // Tracks nodes that have been printed to ensure they are only printed once.
  SmallPtrSet<SymbolUseGraphNode *, 16> done;

  std::function<void(SymbolUseGraphNode *)> printNode = [rootPtr, &printNode, &done,
                                                         &os](SymbolUseGraphNode *node) {
    // Skip if the node has been printed before
    if (!done.insert(node).second) {
      return;
    }
    // Print the current node
    os << "// - Node : [" << node << "] ";
    node->print(os, true, "// --- ");
    // Print list of IDs for the predecessors (excluding root) and successors
    os << "// --- Predecessors : [";
    llvm::interleaveComma(
        llvm::make_filter_range(
            node->predecessorIter(), [rootPtr](SymbolUseGraphNode *n) { return n != rootPtr; }
        ),
        os
    );
    os << "]\n";
    os << "// --- Successors : [";
    llvm::interleaveComma(node->successorIter(), os);
    os << "]\n";
    // Recursively print the successors
    for (SymbolUseGraphNode *c : node->successorIter()) {
      printNode(c);
    }
  };

  os << "// ---- SymbolUseGraph ----\n";
  for (SymbolUseGraphNode *r : rootPtr->successorIter()) {
    printNode(r);
  }
  os << "// ------------------------\n";
  assert(done.size() == this->size() && "All nodes were not printed!");
}

void SymbolUseGraph::dumpToDotFile(std::string filename) const {
  std::string title = llvm::DOTGraphTraits<const llzk::SymbolUseGraph *>::getGraphName(this);
  llvm::WriteGraph(this, "SymbolUseGraph", /*ShortNames*/ false, title, filename);
}

} // namespace llzk
