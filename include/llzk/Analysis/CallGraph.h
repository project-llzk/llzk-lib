//===-- CallGraph.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/Analysis/CallGraph.h>
#include <mlir/Interfaces/CallInterfaces.h>

#include <llvm/ADT/GraphTraits.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/PointerIntPair.h>
#include <llvm/ADT/SCCIterator.h>
#include <llvm/ADT/SetVector.h>

namespace mlir {

class Operation;
class CallOpInterface;
class SymbolTableCollection;

} // namespace mlir

namespace llzk {

/// This is a simple port of the mlir::CallGraphNode with llzk::CallGraph
/// as a friend class, for mlir::CallGraphNode has a private constructor and
/// can only be constructed by mlir::CallGraph. mlir::CallGraphNode is also
/// not polymorphic, so a port is cleaner than requiring casts on mlir::CallGraphNode
/// return types.
class CallGraphNode {
public:
  /// This class represents a directed edge between two nodes in the callgraph.
  class Edge {
    enum class Kind : std::uint8_t {
      // An 'Abstract' edge represents an opaque, non-operation, reference
      // between this node and the target. Edges of this type are only valid
      // from the external node (e.g., an external library call),
      // as there is no valid connection to an operation in the module.
      Abstract,

      // A 'Call' edge represents a direct reference to the target node via a
      // call-like operation within the callable region of this node.
      Call,

      // A 'Child' edge is used when the region of target node is defined inside
      // of the callable region of this node. This means that the region of this
      // node is an ancestor of the region for the target node. As such, this
      // edge cannot be used on the 'external' node.
      Child,
    };

  public:
    /// Returns true if this edge represents an `Abstract` edge.
    bool isAbstract() const { return targetAndKind.getInt() == Kind::Abstract; }

    /// Returns true if this edge represents a `Call` edge.
    bool isCall() const { return targetAndKind.getInt() == Kind::Call; }

    /// Returns true if this edge represents a `Child` edge.
    bool isChild() const { return targetAndKind.getInt() == Kind::Child; }

    /// Returns the source node of this edge.
    /// Note: added by LLZK
    CallGraphNode *getSource() const { return source; }

    /// Returns the target node for this edge.
    CallGraphNode *getTarget() const { return targetAndKind.getPointer(); }

    bool operator==(const Edge &edge) const {
      return source == edge.source && targetAndKind == edge.targetAndKind;
    }

  private:
    Edge(CallGraphNode *src, CallGraphNode *target, Kind kind)
        : source(src), targetAndKind(target, kind) {}
    Edge(CallGraphNode *src, llvm::PointerIntPair<CallGraphNode *, 2, Kind> tgtAndKind)
        : source(src), targetAndKind(tgtAndKind) {}

    /// The source node of this edge.
    /// Note: added by LLZK.
    CallGraphNode *source;

    /// The target node of this edge, as well as the edge kind.
    llvm::PointerIntPair<CallGraphNode *, 2, Kind> targetAndKind;

    // Provide access to the constructor and Kind.
    friend class CallGraphNode;
  };

  /// Returns true if this node is an external node.
  bool isExternal() const;

  /// Returns the callable region this node represents. This can only be called
  /// on non-external nodes.
  mlir::Region *getCallableRegion() const;

  /// Returns the called function that the callable region represents.
  /// As per getCallableRegion, this can only be called on non-external nodes.
  /// This is an LLZK-specific addition.
  mlir::CallableOpInterface getCalledFunction() const;

  /// Adds an abstract reference edge to the given node. An abstract edge does
  /// not come from any observable operations, so this is only valid on the
  /// external node.
  void addAbstractEdge(CallGraphNode *node);

  /// Add an outgoing call edge from this node.
  void addCallEdge(CallGraphNode *node);

  /// Adds a reference edge to the given child node.
  void addChildEdge(CallGraphNode *child);

  /// Iterator over the outgoing edges of this node.
  using iterator = mlir::SmallVectorImpl<Edge>::const_iterator;
  iterator begin() const { return edges.begin(); }
  iterator end() const { return edges.end(); }

  llvm::iterator_range<iterator> edgesOut() const { return llvm::make_range(begin(), end()); }

  /// Returns true if this node has any child edges.
  bool hasChildren() const;

private:
  /// DenseMap info for callgraph edges.
  struct EdgeKeyInfo {
    using SourceInfo = mlir::DenseMapInfo<CallGraphNode *>;
    using BaseInfo = mlir::DenseMapInfo<llvm::PointerIntPair<CallGraphNode *, 2, Edge::Kind>>;

    static Edge getEmptyKey() { return Edge(nullptr, BaseInfo::getEmptyKey()); }
    static Edge getTombstoneKey() { return Edge(nullptr, BaseInfo::getTombstoneKey()); }
    static unsigned getHashValue(const Edge &edge) {
      return SourceInfo::getHashValue(edge.source) ^ BaseInfo::getHashValue(edge.targetAndKind);
    }
    static bool isEqual(const Edge &lhs, const Edge &rhs) { return lhs == rhs; }
  };

  CallGraphNode(mlir::Region *callable) : callableRegion(callable) {}

  /// Add an edge to 'node' with the given kind.
  void addEdge(CallGraphNode *node, Edge::Kind kind);

  /// The callable region defines the boundary of the call graph node. This is
  /// the region referenced by 'call' operations. This is at a per-region
  /// boundary as operations may define multiple callable regions.
  mlir::Region *callableRegion;

  /// A set of out-going edges from this node to other nodes in the graph.
  mlir::SetVector<Edge, mlir::SmallVector<Edge, 4>, llvm::SmallDenseSet<Edge, 4, EdgeKeyInfo>>
      edges;

  // Provide access to private methods.
  friend class CallGraph;
};

/// This is a port of mlir::CallGraph that has been adapted to use the custom
/// symbol lookup helpers (see SymbolHelper.h). Unfortunately the mlir::CallGraph
/// is not readily extensible, so we will define our own with a similar interface.
class CallGraph {
  using NodeMapT = llvm::MapVector<mlir::Region *, std::unique_ptr<CallGraphNode>>;

  /// This class represents an iterator over the internal call graph nodes. This
  /// class unwraps the map iterator to access the raw node.
  class NodeIterator final
      : public llvm::mapped_iterator<
            NodeMapT::const_iterator, CallGraphNode *(*)(const NodeMapT::value_type &)> {
    static CallGraphNode *unwrap(const NodeMapT::value_type &value) { return value.second.get(); }

  public:
    /// Initializes the result type iterator to the specified result iterator.
    NodeIterator(NodeMapT::const_iterator it)
        : llvm::mapped_iterator<
              NodeMapT::const_iterator, CallGraphNode *(*)(const NodeMapT::value_type &)>(
              it, &unwrap
          ) {}
  };

public:
  CallGraph(mlir::Operation *op);

  /// Get or add a call graph node for the given region. `parentNode`
  /// corresponds to the direct node in the callgraph that contains the parent
  /// operation of `region`, or nullptr if there is no parent node.
  CallGraphNode *getOrAddNode(mlir::Region *region, CallGraphNode *parentNode);

  /// Lookup a call graph node for the given region, or nullptr if none is
  /// registered.
  CallGraphNode *lookupNode(mlir::Region *region) const;

  /// Return the callgraph node representing an external caller.
  CallGraphNode *getExternalCallerNode() const {
    return const_cast<CallGraphNode *>(&externalCallerNode);
  }

  /// Return the callgraph node representing an indirect callee.
  CallGraphNode *getUnknownCalleeNode() const {
    return const_cast<CallGraphNode *>(&unknownCalleeNode);
  }

  /// Resolve the callable for given callee to a node in the callgraph, or the
  /// external node if a valid node was not resolved. The provided symbol table
  /// is used when resolving calls that reference callables via a symbol
  /// reference.
  CallGraphNode *
  resolveCallable(mlir::CallOpInterface call, mlir::SymbolTableCollection &symbolTable) const;

  /// Erase the given node from the callgraph.
  void eraseNode(CallGraphNode *node);

  /// An iterator over the nodes of the graph.
  using iterator = NodeIterator;
  iterator begin() const { return nodes.begin(); }
  iterator end() const { return nodes.end(); }

  size_t size() const { return nodes.size(); }

  /// Dump the graph in a human readable format.
  void dump() const;
  void print(llvm::raw_ostream &os) const;

private:
  /// The set of nodes within the callgraph.
  NodeMapT nodes;

  /// A special node used to indicate an external caller.
  CallGraphNode externalCallerNode;

  /// A special node used to indicate an unknown callee.
  CallGraphNode unknownCalleeNode;
};

} // namespace llzk

namespace llvm {
// Provide graph traits for traversing call graphs using standard graph
// traversals.
template <> struct GraphTraits<const llzk::CallGraphNode *> {
  using NodeRef = const llzk::CallGraphNode *;
  static NodeRef getEntryNode(NodeRef node) { return node; }

  static NodeRef unwrap(const llzk::CallGraphNode::Edge &edge) { return edge.getTarget(); }

  // ChildIteratorType/begin/end - Allow iteration over all nodes in the graph.
  using ChildIteratorType = mapped_iterator<llzk::CallGraphNode::iterator, decltype(&unwrap)>;
  static ChildIteratorType child_begin(NodeRef node) { return {node->begin(), &unwrap}; }
  static ChildIteratorType child_end(NodeRef node) { return {node->end(), &unwrap}; }
};

template <>
struct GraphTraits<const llzk::CallGraph *> : public GraphTraits<const llzk::CallGraphNode *> {
  /// The entry node into the graph is the external node.
  static NodeRef getEntryNode(const llzk::CallGraph *cg) { return cg->getExternalCallerNode(); }

  // nodes_iterator/begin/end - Allow iteration over all nodes in the graph
  using nodes_iterator = llzk::CallGraph::iterator;
  static nodes_iterator nodes_begin(llzk::CallGraph *cg) { return cg->begin(); }
  static nodes_iterator nodes_end(llzk::CallGraph *cg) { return cg->end(); }
};
} // namespace llvm
