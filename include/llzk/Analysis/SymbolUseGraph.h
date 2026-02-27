//===-- SymbolUseGraph.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/DOTGraphTraits.h>

#include <utility>

namespace llzk {

class SymbolUseGraphNode {
  using OpSet = llvm::SmallPtrSet<mlir::Operation *, 3>;

  mlir::ModuleOp symbolPathRoot;
  mlir::SymbolRefAttr symbolPath;
  OpSet opsThatUseTheSymbol;
  bool isStructConstParam;

  /* Tree structure. The SymbolUseGraph owns the nodes so just pointers here. */
  /// Predecessor: Symbol that uses the current Symbol with its defining Operation.
  mlir::SetVector<SymbolUseGraphNode *> predecessors;
  /// Successor: Symbol that is used within the current Symbol defining Operation.
  mlir::SetVector<SymbolUseGraphNode *> successors;

  SymbolUseGraphNode(mlir::ModuleOp pathRoot, mlir::SymbolRefAttr path)
      : symbolPathRoot(pathRoot), symbolPath(path), isStructConstParam(false) {
    assert(pathRoot && "'pathRoot' cannot be nullptr");
    assert(path && "'path' cannot be nullptr");
  }

  /// Used only for creating the artificial root/head and tail nodes in the graph.
  SymbolUseGraphNode() : symbolPathRoot(nullptr), symbolPath(nullptr), isStructConstParam(false) {}

  /// Return 'false' iff the given node is an artificial node created for the graph head/tail.
  static bool isRealNodeImpl(const SymbolUseGraphNode *node) { return node->symbolPath != nullptr; }

  /// Add a successor node.
  void addSuccessor(SymbolUseGraphNode *node);

  /// Remove a successor node.
  void removeSuccessor(SymbolUseGraphNode *node);

  // Provide access to private members.
  friend class SymbolUseGraph;

public:
  /// Return 'false' iff this node is an artificial node created for the graph head/tail.
  /// This type of node should only be returned via the GraphTraits.
  bool isRealNode() const { return isRealNodeImpl(this); }

  /// Return the root ModuleOp for the path.
  mlir::ModuleOp getSymbolPathRoot() const { return symbolPathRoot; }

  /// The symbol path+name relative to the closest root ModuleOp.
  mlir::SymbolRefAttr getSymbolPath() const { return symbolPath; }

  /// The set of operations that use the symbol.
  const OpSet &getUserOps() const { return opsThatUseTheSymbol; }

  /// Return `true` iff the symbol is a struct constant parameter name.
  bool isStructParam() const { return isStructConstParam; }

  /// Return true if this node has any predecessors.
  bool hasPredecessor() const {
    return llvm::find_if(predecessors, isRealNodeImpl) != predecessors.end();
  }
  size_t numPredecessors() const { return llvm::count_if(predecessors, isRealNodeImpl); }

  /// Return true if this node has any successors.
  bool hasSuccessor() const {
    return llvm::find_if(successors, isRealNodeImpl) != successors.end();
  }
  size_t numSuccessors() const { return llvm::count_if(successors, isRealNodeImpl); }

  /// Iterator over predecessors/successors.
  using iterator = llvm::filter_iterator<
      mlir::SetVector<SymbolUseGraphNode *>::const_iterator, bool (*)(const SymbolUseGraphNode *)>;

  inline iterator predecessors_begin() const { return predecessorIter().begin(); }
  inline iterator predecessors_end() const { return predecessorIter().end(); }
  inline iterator successors_begin() const { return successorIter().begin(); }
  inline iterator successors_end() const { return successorIter().end(); }

  /// Range over predecessor nodes.
  llvm::iterator_range<iterator> predecessorIter() const {
    return llvm::make_filter_range(predecessors, isRealNodeImpl);
  }

  /// Range over successor nodes.
  llvm::iterator_range<iterator> successorIter() const {
    return llvm::make_filter_range(successors, isRealNodeImpl);
  }

  mlir::FailureOr<SymbolLookupResultUntyped>
  lookupSymbol(mlir::SymbolTableCollection &tables, bool reportMissing = true) const;

  /// Print the node in a human readable format.
  std::string toString(bool showLocations = false) const;
  void print(
      llvm::raw_ostream &os, bool showLocations = false, const std::string &locationLinePrefix = ""
  ) const;
};

/// Builds a graph structure representing the relationships between symbols and their uses. There is
/// a node for each SymbolRef and the successors are the Symbols uses within this Symbol's defining
/// Operation.
class SymbolUseGraph {
  using NodeMapKeyT = std::pair<mlir::ModuleOp, mlir::SymbolRefAttr>;
  /// Maps Symbol operation to the (owned) SymbolUseGraphNode for that op
  using NodeMapT = llvm::MapVector<NodeMapKeyT, std::unique_ptr<SymbolUseGraphNode>>;

  /// The set of nodes within the graph.
  NodeMapT nodes;

  // The singleton artificial (i.e., no associated op) root/head and tail nodes of the graph. Every
  // newly created SymbolUseGraphNode is initially a successor of the root node until a real
  // successor (if any) is added. Similarly, all leaf nodes in the graph have the tail as successor.
  //
  // Implementation note: An actual SymbolUseGraphNode is used instead of lists of head/tail nodes
  // because the GraphTraits implementations require a single entry node. These nodes are not added
  // to the `nodes` set and should be transparent to users of this graph (other than through the
  // GraphTraits `getEntryNode()` function implementations).
  SymbolUseGraphNode root, tail;

  /// An iterator over the internal graph nodes. Unwraps the map iterator to access the node.
  class NodeIterator final
      : public llvm::mapped_iterator<
            NodeMapT::const_iterator, SymbolUseGraphNode *(*)(const NodeMapT::value_type &)> {
    static SymbolUseGraphNode *unwrap(const NodeMapT::value_type &value) {
      return value.second.get();
    }

  public:
    /// Initializes the result type iterator to the specified result iterator.
    NodeIterator(NodeMapT::const_iterator it)
        : llvm::mapped_iterator<
              NodeMapT::const_iterator, SymbolUseGraphNode *(*)(const NodeMapT::value_type &)>(
              it, &unwrap
          ) {}
  };

  /// Get or add a graph node for the given symbol reference relative to the given root ModuleOp.
  SymbolUseGraphNode *getOrAddNode(
      mlir::ModuleOp pathRoot, mlir::SymbolRefAttr path, SymbolUseGraphNode *predecessorNode
  );

  SymbolUseGraphNode *getSymbolUserNode(const mlir::SymbolTable::SymbolUse &u);
  void buildGraph(mlir::SymbolOpInterface symbolOp);

  // Friend declarations for the specializations of GraphTraits
  friend struct llvm::GraphTraits<const llzk::SymbolUseGraph *>;
  friend struct llvm::GraphTraits<llvm::Inverse<const llzk::SymbolUseGraph *>>;

public:
  SymbolUseGraph(mlir::SymbolOpInterface rootSymbolOp);

  /// Return the existing node for the symbol reference relative to the given module, else nullptr.
  const SymbolUseGraphNode *lookupNode(mlir::ModuleOp pathRoot, mlir::SymbolRefAttr path) const;

  /// Return the existing node for the symbol definition op, else nullptr.
  const SymbolUseGraphNode *lookupNode(mlir::SymbolOpInterface symbolDef) const;

  /// Return the total number of nodes in the graph.
  size_t size() const { return nodes.size(); }

  /// Iterator over the root nodes (i.e., nodes that have no predecessors).
  using roots_iterator = SymbolUseGraphNode::iterator;
  roots_iterator roots_begin() const { return root.successors_begin(); }
  roots_iterator roots_end() const { return root.successors_end(); }

  /// Range over root nodes (i.e., nodes that have no predecessors).
  inline llvm::iterator_range<roots_iterator> rootsIter() const {
    return llvm::make_range(roots_begin(), roots_end());
  }

  /// An iterator over the nodes of the graph.
  using iterator = NodeIterator;
  iterator begin() const { return nodes.begin(); }
  iterator end() const { return nodes.end(); }

  /// Range over all nodes in the graph.
  inline llvm::iterator_range<iterator> nodesIter() const {
    return llvm::make_range(begin(), end());
  }

  /// Dump the graph in a human readable format.
  inline void dump() const { print(llvm::errs()); }
  void print(llvm::raw_ostream &os) const;

  /// Dump the graph to file in dot graph format.
  void dumpToDotFile(std::string filename = "") const;
};

} // namespace llzk

namespace llvm {

// Provide graph traits for traversing SymbolUseGraph using standard graph traversals.
template <> struct GraphTraits<const llzk::SymbolUseGraphNode *> {
  using NodeRef = const llzk::SymbolUseGraphNode *;
  static NodeRef getEntryNode(NodeRef node) { return node; }

  /// ChildIteratorType/begin/end - Allow iteration over all nodes in the graph.
  using ChildIteratorType = llzk::SymbolUseGraphNode::iterator;
  static ChildIteratorType child_begin(NodeRef node) { return node->successors_begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node->successors_end(); }
};

template <>
struct GraphTraits<const llzk::SymbolUseGraph *>
    : public GraphTraits<const llzk::SymbolUseGraphNode *> {
  using GraphType = const llzk::SymbolUseGraph *;

  /// The entry node into the graph is the root node.
  static NodeRef getEntryNode(GraphType g) { return &g->root; }

  /// nodes_iterator/begin/end - Allow iteration over all nodes in the graph.
  using nodes_iterator = llzk::SymbolUseGraph::iterator;
  static nodes_iterator nodes_begin(GraphType g) { return g->begin(); }
  static nodes_iterator nodes_end(GraphType g) { return g->end(); }

  /// Return total number of nodes in the graph.
  static unsigned size(GraphType g) { return g->size(); }
};

// Provide graph traits for traversing SymbolUseGraph using INVERSE graph traversals.
template <> struct GraphTraits<Inverse<const llzk::SymbolUseGraphNode *>> {
  using NodeRef = const llzk::SymbolUseGraphNode *;
  static NodeRef getEntryNode(Inverse<NodeRef> node) { return node.Graph; }

  using ChildIteratorType = llzk::SymbolUseGraphNode::iterator;
  static ChildIteratorType child_begin(NodeRef node) { return node->predecessors_begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node->predecessors_end(); }
};

template <>
struct GraphTraits<Inverse<const llzk::SymbolUseGraph *>>
    : public GraphTraits<Inverse<const llzk::SymbolUseGraphNode *>> {
  using GraphType = Inverse<const llzk::SymbolUseGraph *>;

  /// The entry node into the inverse graph is the tail node.
  static NodeRef getEntryNode(GraphType g) { return &g.Graph->tail; }

  /// nodes_iterator/begin/end - Allow iteration over all nodes in the graph.
  using nodes_iterator = llzk::SymbolUseGraph::iterator;
  static nodes_iterator nodes_begin(GraphType g) { return g.Graph->begin(); }
  static nodes_iterator nodes_end(GraphType g) { return g.Graph->end(); }

  /// Return total number of nodes in the graph.
  static unsigned size(GraphType g) { return g.Graph->size(); }
};

// Provide graph traits for printing SymbolUseGraph using dot graph printer.
template <> struct DOTGraphTraits<const llzk::SymbolUseGraphNode *> : public DefaultDOTGraphTraits {
  using NodeRef = const llzk::SymbolUseGraphNode *;
  using GraphType = const llzk::SymbolUseGraph *;

  DOTGraphTraits(bool isSimple = false) : DefaultDOTGraphTraits(isSimple) {}

  std::string getNodeLabel(NodeRef n, GraphType) { return n->toString(true); }
};

template <>
struct DOTGraphTraits<const llzk::SymbolUseGraph *>
    : public DOTGraphTraits<const llzk::SymbolUseGraphNode *> {

  DOTGraphTraits(bool isSimple = false) : DOTGraphTraits<NodeRef>(isSimple) {}

  static std::string getGraphName(GraphType) { return "Symbol Use Graph"; }

  std::string getNodeLabel(NodeRef n, GraphType g) {
    return DOTGraphTraits<NodeRef>::getNodeLabel(n, g);
  }
};

} // namespace llvm
