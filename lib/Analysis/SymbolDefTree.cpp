//===-- SymbolDefTree.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SymbolDefTree.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DepthFirstIterator.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/Support/GraphWriter.h>

using namespace mlir;

namespace llzk {

//===----------------------------------------------------------------------===//
// SymbolDefTreeNode
//===----------------------------------------------------------------------===//

void SymbolDefTreeNode::addChild(SymbolDefTreeNode *node) {
  assert(!node->parent && "def cannot be in more than one symbol table");
  node->parent = this;
  children.insert(node);
}

//===----------------------------------------------------------------------===//
// SymbolDefTree
//===----------------------------------------------------------------------===//

namespace {

void assertProperBuild(SymbolOpInterface root, const SymbolDefTree *tree) {
  // Collect all Symbols in the graph
  llvm::SmallSet<SymbolOpInterface, 16> fromGraph;
  for (const SymbolDefTreeNode *r : llvm::depth_first(tree)) {
    if (SymbolOpInterface s = r->getOp()) {
      fromGraph.insert(s);
    }
  }
  // Ensure every symbol reachable from the 'root' is represented in the graph
#ifndef NDEBUG
  root.walk([&fromGraph](SymbolOpInterface s) { assert(fromGraph.contains(s)); });
#endif
}

} // namespace

SymbolDefTree::SymbolDefTree(SymbolOpInterface rootSymbol) {
  assert(rootSymbol->hasTrait<OpTrait::SymbolTable>());
  buildTree(rootSymbol, /*parentNode=*/nullptr);
  assertProperBuild(rootSymbol, this);
}

void SymbolDefTree::buildTree(SymbolOpInterface symbolOp, SymbolDefTreeNode *parentNode) {
  // Add node for the current symbol
  parentNode = getOrAddNode(symbolOp, parentNode);
  // If this symbol is also its own SymbolTable, recursively add child symbols
  if (symbolOp->hasTrait<OpTrait::SymbolTable>()) {
    for (Operation &op : symbolOp->getRegion(0).front()) {
      if (SymbolOpInterface childSym = llvm::dyn_cast<SymbolOpInterface>(&op)) {
        buildTree(childSym, parentNode);
      }
    }
  }
}

SymbolDefTreeNode *
SymbolDefTree::getOrAddNode(SymbolOpInterface symbolDef, SymbolDefTreeNode *parentNode) {
  std::unique_ptr<SymbolDefTreeNode> &node = nodes[symbolDef];
  if (!node) {
    node.reset(new SymbolDefTreeNode(symbolDef));
    // Add this node to the given parent node if given, else the root node.
    if (parentNode) {
      parentNode->addChild(node.get());
    } else {
      root.addChild(node.get());
    }
  }
  return node.get();
}

const SymbolDefTreeNode *SymbolDefTree::lookupNode(SymbolOpInterface symbolDef) const {
  const auto *it = nodes.find(symbolDef);
  return it == nodes.end() ? nullptr : it->second.get();
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

std::string SymbolDefTreeNode::toString() const { return buildStringViaPrint(*this); }

void SymbolDefTreeNode::print(llvm::raw_ostream &os) const {
  os << '\'' << symbolDef->getName() << "' ";
  if (StringAttr name = llzk::getSymbolName(symbolDef)) {
    os << "named " << name << '\n';
  } else {
    os << "without a name\n";
  }
}

void SymbolDefTree::print(llvm::raw_ostream &os) const {
  std::function<void(SymbolDefTreeNode *)> printNode = [&os, &printNode](SymbolDefTreeNode *node) {
    // Print the current node
    os << "// - Node : [" << node << "] ";
    node->print(os);
    // Print list of IDs for the children
    os << "// --- Children : [";
    llvm::interleaveComma(node->children, os);
    os << "]\n";
    // Recursively print the children
    for (SymbolDefTreeNode *c : node->children) {
      printNode(c);
    }
  };

  os << "// ---- SymbolDefTree ----\n";
  for (SymbolDefTreeNode *r : root.children) {
    printNode(r);
  }
  os << "// -----------------------\n";
}

void SymbolDefTree::dumpToDotFile(std::string filename) const {
  std::string title = llvm::DOTGraphTraits<const llzk::SymbolDefTree *>::getGraphName(this);
  llvm::WriteGraph(this, "SymbolDefTree", /*ShortNames*/ false, title, filename);
}

} // namespace llzk
