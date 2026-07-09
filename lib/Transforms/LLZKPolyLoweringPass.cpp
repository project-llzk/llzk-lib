//===-- LLZKPolyLoweringPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-poly-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>
#include <optional>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_POLYLOWERINGPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;
using namespace llzk::array;

#define DEBUG_TYPE "llzk-poly-lowering-pass"
#define AUXILIARY_MEMBER_PREFIX "__llzk_poly_lowering_pass_aux_member_"

namespace {

struct AuxAssignment {
  std::string auxMemberName;
  Value computedValue;
  Value auxValue;
};

/// Tracks a mutable felt-array element operand together with its relative index.
struct MutableContainmentElement {
  ArrayAttr index;
  OpOperand *operand;
};

enum class AuxAssignmentVisitState : uint8_t {
  Unvisited,
  Visiting,
  Done,
};

class PassImpl : public llzk::impl::PolyLoweringPassBase<PassImpl> {
  using Base = PolyLoweringPassBase<PassImpl>;
  using Base::Base;

  unsigned auxCounter = 0;

  void collectStructDefs(ModuleOp modOp, SmallVectorImpl<StructDefOp> &structDefs) {
    modOp.walk([&structDefs](StructDefOp structDef) {
      structDefs.push_back(structDef);
      return WalkResult::skip();
    });
  }

  /// Records a dependency from the current aux assignment to a prerequisite.
  void addAuxDependency(
      unsigned dep, unsigned owner, DenseSet<unsigned> &seenDeps, SmallVectorImpl<unsigned> &deps
  ) const {
    if (dep == owner) {
      return;
    }
    if (seenDeps.insert(dep).second) {
      deps.push_back(dep);
    }
  }

  /// Collects aux assignments that must be written before the given value can be rebuilt.
  void collectAuxDependencies(
      Value val, unsigned owner, const DenseMap<Value, unsigned> &auxValueToIndex,
      const llvm::StringMap<unsigned> &auxNameToIndex, DenseSet<Value> &visitedValues,
      DenseSet<unsigned> &seenDeps, SmallVectorImpl<unsigned> &deps
  ) const {
    // Aux dependencies can appear as generated aux SSA values or reads of generated
    // aux members, so track both forms before ordering writes.
    if (!val || !visitedValues.insert(val).second) {
      return;
    }

    if (auto it = auxValueToIndex.find(val); it != auxValueToIndex.end()) {
      addAuxDependency(it->second, owner, seenDeps, deps);
    }

    if (auto readOp = val.getDefiningOp<MemberReadOp>()) {
      auto it = auxNameToIndex.find(readOp.getMemberName());
      if (it != auxNameToIndex.end()) {
        addAuxDependency(it->second, owner, seenDeps, deps);
      }
    }

    if (Operation *defOp = val.getDefiningOp()) {
      for (Value operand : defOp->getOperands()) {
        collectAuxDependencies(
            operand, owner, auxValueToIndex, auxNameToIndex, visitedValues, seenDeps, deps
        );
      }
    }
  }

  /// Visits aux assignments depth-first so dependencies are emitted before users.
  LogicalResult visitAuxAssignment(
      unsigned idx, ArrayRef<SmallVector<unsigned>> deps,
      SmallVectorImpl<AuxAssignmentVisitState> &visitState, SmallVectorImpl<unsigned> &ordered,
      ArrayRef<AuxAssignment> auxAssignments
  ) const {
    if (visitState[idx] == AuxAssignmentVisitState::Done) {
      return success();
    }
    if (visitState[idx] == AuxAssignmentVisitState::Visiting) {
      return emitError(auxAssignments[idx].computedValue.getLoc())
             << "poly lowering generated cyclic auxiliary dependency involving @"
             << auxAssignments[idx].auxMemberName;
    }

    visitState[idx] = AuxAssignmentVisitState::Visiting;
    // Emit prerequisite aux writes before the aux writes that read them.
    for (unsigned dep : deps[idx]) {
      if (failed(visitAuxAssignment(dep, deps, visitState, ordered, auxAssignments))) {
        return failure();
      }
    }
    visitState[idx] = AuxAssignmentVisitState::Done;
    ordered.push_back(idx);
    return success();
  }

  /// Produces a topological write order for generated aux assignments.
  LogicalResult orderAuxAssignments(
      ArrayRef<AuxAssignment> auxAssignments, SmallVectorImpl<unsigned> &ordered
  ) const {
    DenseMap<Value, unsigned> auxValueToIndex;
    llvm::StringMap<unsigned> auxNameToIndex;
    auxValueToIndex.reserve(auxAssignments.size());
    for (auto [idx, assign] : llvm::enumerate(auxAssignments)) {
      if (assign.auxValue) {
        auxValueToIndex[assign.auxValue] = idx;
      }
      auxNameToIndex[assign.auxMemberName] = idx;
    }

    SmallVector<SmallVector<unsigned>> deps(auxAssignments.size());
    for (auto [idx, assign] : llvm::enumerate(auxAssignments)) {
      DenseSet<Value> visitedValues;
      DenseSet<unsigned> seenDeps;
      collectAuxDependencies(
          assign.computedValue, idx, auxValueToIndex, auxNameToIndex, visitedValues, seenDeps,
          deps[idx]
      );
    }

    SmallVector<AuxAssignmentVisitState> visitState(
        auxAssignments.size(), AuxAssignmentVisitState::Unvisited
    );
    for (unsigned idx = 0, e = auxAssignments.size(); idx < e; ++idx) {
      if (failed(visitAuxAssignment(idx, deps, visitState, ordered, auxAssignments))) {
        return failure();
      }
    }
    return success();
  }

  // Recursively compute degree of FeltOps SSA values
  unsigned getDegree(Value val, DenseMap<Value, unsigned> &memo) {
    if (auto it = memo.find(val); it != memo.end()) {
      return it->second;
    }
    // Handle function parameters (BlockArguments)
    if (llvm::isa<BlockArgument>(val)) {
      memo[val] = 1;
      return 1;
    }
    if (val.getDefiningOp<FeltConstantOp>()) {
      return memo[val] = 0;
    }
    if (val.getDefiningOp<NonDetOp>()) {
      return memo[val] = 1;
    }
    if (val.getDefiningOp<MemberReadOp>()) {
      return memo[val] = 1;
    }
    if (auto addOp = val.getDefiningOp<AddFeltOp>()) {
      return memo[val] = std::max(getDegree(addOp.getLhs(), memo), getDegree(addOp.getRhs(), memo));
    }
    if (auto subOp = val.getDefiningOp<SubFeltOp>()) {
      return memo[val] = std::max(getDegree(subOp.getLhs(), memo), getDegree(subOp.getRhs(), memo));
    }
    if (auto mulOp = val.getDefiningOp<MulFeltOp>()) {
      return memo[val] = getDegree(mulOp.getLhs(), memo) + getDegree(mulOp.getRhs(), memo);
    }
    if (auto divOp = val.getDefiningOp<DivFeltOp>()) {
      return memo[val] = getDegree(divOp.getLhs(), memo) + getDegree(divOp.getRhs(), memo);
    }
    if (auto negOp = val.getDefiningOp<NegFeltOp>()) {
      return memo[val] = getDegree(negOp.getOperand(), memo);
    }

    llvm_unreachable("Unhandled Felt SSA value in degree computation");
  }

  Value lowerExpression(
      Value val, StructDefOp structDef, FuncDefOp constrainFunc, Operation *useOp,
      DominanceInfo &dominanceInfo, DenseMap<Value, unsigned> &degreeMemo,
      DenseMap<Value, Value> &rewrites, SmallVector<AuxAssignment> &auxAssignments
  ) {
    auto rewriteIt = rewrites.find(val);
    if (rewriteIt != rewrites.end() && dominanceInfo.properlyDominates(rewriteIt->second, useOp)) {
      return rewriteIt->second;
    }

    auto cacheIdentityRewriteIfAbsent = [&rewrites, &val]() {
      // Keep an existing cached aux replacement for later uses it may dominate.
      if (!rewrites.contains(val)) {
        rewrites[val] = val;
      }
    };

    unsigned degree = getDegree(val, degreeMemo);
    if (degree <= maxDegree) {
      // A cached replacement that does not dominate this use may still be the
      // right replacement for later uses. Return the original value for this
      // use without clobbering that scoped rewrite.
      cacheIdentityRewriteIfAbsent();
      return val;
    }

    // Degree-neutral roots can still contain over-degree operands.
    auto lowerBinaryRoot = [&](auto op) -> Value {
      Value lhs = lowerExpression(
          op.getLhs(), structDef, constrainFunc, op.getOperation(), dominanceInfo, degreeMemo,
          rewrites, auxAssignments
      );
      Value rhs = lowerExpression(
          op.getRhs(), structDef, constrainFunc, op.getOperation(), dominanceInfo, degreeMemo,
          rewrites, auxAssignments
      );

      if (lhs != op.getLhs()) {
        op.getLhsMutable().set(lhs);
      }
      if (rhs != op.getRhs()) {
        op.getRhsMutable().set(rhs);
      }
      degreeMemo[val] = std::max(getDegree(lhs, degreeMemo), getDegree(rhs, degreeMemo));
      cacheIdentityRewriteIfAbsent();
      return val;
    };

    if (auto addOp = val.getDefiningOp<AddFeltOp>()) {
      return lowerBinaryRoot(addOp);
    }

    if (auto subOp = val.getDefiningOp<SubFeltOp>()) {
      return lowerBinaryRoot(subOp);
    }

    if (auto negOp = val.getDefiningOp<NegFeltOp>()) {
      Value operand = lowerExpression(
          negOp.getOperand(), structDef, constrainFunc, negOp.getOperation(), dominanceInfo,
          degreeMemo, rewrites, auxAssignments
      );

      if (operand != negOp.getOperand()) {
        negOp.getOperandMutable().set(operand);
      }
      degreeMemo[val] = getDegree(operand, degreeMemo);
      cacheIdentityRewriteIfAbsent();
      return val;
    }

    if (auto mulOp = val.getDefiningOp<MulFeltOp>()) {
      // Recursively lower operands first
      Value lhs = lowerExpression(
          mulOp.getLhs(), structDef, constrainFunc, mulOp.getOperation(), dominanceInfo, degreeMemo,
          rewrites, auxAssignments
      );
      Value rhs = lowerExpression(
          mulOp.getRhs(), structDef, constrainFunc, mulOp.getOperation(), dominanceInfo, degreeMemo,
          rewrites, auxAssignments
      );

      unsigned lhsDeg = getDegree(lhs, degreeMemo);
      unsigned rhsDeg = getDegree(rhs, degreeMemo);

      OpBuilder builder(mulOp.getOperation()->getBlock(), ++Block::iterator(mulOp));
      Value selfVal = constrainFunc.getSelfValueFromConstrain();
      bool eraseMul = lhsDeg + rhsDeg > maxDegree;
      // Optimization: If lhs == rhs, factor it only once
      if (lhs == rhs && eraseMul) {
        std::string auxName = AUXILIARY_MEMBER_PREFIX + std::to_string(this->auxCounter++);
        MemberDefOp auxMember = addAuxMember(structDef, auxName, lhs.getType());

        auto auxVal = builder.create<MemberReadOp>(
            lhs.getLoc(), lhs.getType(), selfVal, auxMember.getNameAttr()
        );
        auxAssignments.push_back({auxName, lhs, auxVal});
        Location loc = builder.getFusedLoc({auxVal.getLoc(), lhs.getLoc()});
        auto eqOp = builder.create<EmitEqualityOp>(loc, auxVal, lhs);

        // Memoize auxVal as degree 1
        degreeMemo[auxVal] = 1;
        rewrites[lhs] = auxVal;
        rewrites[rhs] = auxVal;
        // Now selectively replace subsequent uses of lhs with auxVal
        replaceSubsequentUsesWith(lhs, auxVal, eqOp);

        // Update lhs and rhs to use auxVal
        lhs = auxVal;
        rhs = auxVal;

        lhsDeg = rhsDeg = 1;
      }
      // While their product exceeds maxDegree, factor out one side
      while (lhsDeg + rhsDeg > maxDegree) {
        Value &toFactor = (lhsDeg >= rhsDeg) ? lhs : rhs;

        // Create auxiliary member for toFactor
        std::string auxName = AUXILIARY_MEMBER_PREFIX + std::to_string(this->auxCounter++);
        MemberDefOp auxMember = addAuxMember(structDef, auxName, toFactor.getType());

        // Read back as MemberReadOp (new SSA value)
        auto auxVal = builder.create<MemberReadOp>(
            toFactor.getLoc(), toFactor.getType(), selfVal, auxMember.getNameAttr()
        );

        // Emit constraint: auxVal == toFactor
        Location loc = builder.getFusedLoc({auxVal.getLoc(), toFactor.getLoc()});
        auto eqOp = builder.create<EmitEqualityOp>(loc, auxVal, toFactor);
        auxAssignments.push_back({auxName, toFactor, auxVal});
        // Update memoization
        rewrites[toFactor] = auxVal;
        degreeMemo[auxVal] = 1; // stays same
        // replace the term with auxVal.
        replaceSubsequentUsesWith(toFactor, auxVal, eqOp);

        // Remap toFactor to auxVal for next iterations
        toFactor = auxVal;

        // Recompute degrees
        lhsDeg = getDegree(lhs, degreeMemo);
        rhsDeg = getDegree(rhs, degreeMemo);
      }

      // Now lhs * rhs fits within degree bound
      auto mulVal = builder.create<MulFeltOp>(lhs.getLoc(), lhs.getType(), lhs, rhs);
      if (eraseMul) {
        mulOp->replaceAllUsesWith(mulVal);
        mulOp->erase();
      }

      // Result of this multiply has degree lhsDeg + rhsDeg
      degreeMemo[mulVal] = lhsDeg + rhsDeg;
      rewrites[val] = mulVal;

      return mulVal;
    }

    // Unsupported roots are left unchanged.
    cacheIdentityRewriteIfAbsent();
    return val;
  }

  Value materializeCallArgument(
      Value val, StructDefOp structDef, FuncDefOp constrainFunc, CallOp callOp,
      DominanceInfo &dominanceInfo, DenseMap<Value, unsigned> &degreeMemo,
      DenseMap<Value, Value> &rewrites, SmallVector<AuxAssignment> &auxAssignments
  ) {
    Value loweredVal = lowerExpression(
        val, structDef, constrainFunc, callOp.getOperation(), dominanceInfo, degreeMemo, rewrites,
        auxAssignments
    );
    DenseMap<Value, unsigned> checkMemo;
    if (getDegree(loweredVal, checkMemo) <= 1) {
      return loweredVal;
    }

    // Callees only receive SSA values, not the caller expression tree, so nonlinear
    // call arguments must be represented by an auxiliary member read.
    std::string auxName = AUXILIARY_MEMBER_PREFIX + std::to_string(this->auxCounter++);
    MemberDefOp auxMember = addAuxMember(structDef, auxName, loweredVal.getType());

    OpBuilder builder(callOp);
    Value selfVal = constrainFunc.getSelfValueFromConstrain();
    auto auxVal = builder.create<MemberReadOp>(
        loweredVal.getLoc(), loweredVal.getType(), selfVal, auxMember.getNameAttr()
    );

    Location loc = builder.getFusedLoc({auxVal.getLoc(), loweredVal.getLoc()});
    builder.create<EmitEqualityOp>(loc, auxVal, loweredVal);
    auxAssignments.push_back({auxName, loweredVal, auxVal});

    degreeMemo[auxVal] = 1;
    rewrites[loweredVal] = auxVal;
    rewrites[val] = auxVal;
    return auxVal;
  }

  LogicalResult checkEqualityDegrees(FuncDefOp constrainFunc) {
    bool failedCheck = false;

    constrainFunc.walk([&](EmitEqualityOp eqOp) {
      DenseMap<Value, unsigned> checkMemo;
      unsigned lhsDegree = getDegree(eqOp.getLhs(), checkMemo);
      unsigned rhsDegree = getDegree(eqOp.getRhs(), checkMemo);

      if (lhsDegree > maxDegree || rhsDegree > maxDegree) {
        auto diag = eqOp.emitOpError();
        diag << "poly lowering postcondition failed: equality operand degree exceeds max-degree "
             << maxDegree.getValue() << " (lhs degree " << lhsDegree << ", rhs degree " << rhsDegree
             << ")";
        diag.report();
        failedCheck = true;
      }
    });

    return failure(failedCheck);
  }

  LogicalResult checkStructConstrainCallArguments(FuncDefOp constrainFunc) {
    bool failedCheck = false;

    constrainFunc.walk([&](CallOp callOp) {
      if (!callOp.calleeIsStructConstrain()) {
        return;
      }

      for (Value arg : callOp.getArgOperands()) {
        if (!llvm::isa<FeltType>(arg.getType())) {
          continue;
        }

        DenseMap<Value, unsigned> checkMemo;
        unsigned argDegree = getDegree(arg, checkMemo);
        if (argDegree > 1) {
          auto diag = callOp.emitOpError();
          diag << "poly lowering postcondition failed: struct constrain call argument degree "
                  "exceeds 1 (argument degree "
               << argDegree << ")";
          diag.report();
          failedCheck = true;
        }
      }
    });

    return failure(failedCheck);
  }

  /// Returns true when \p type is an array whose element type is FeltType.
  bool isFeltArray(Type type) const {
    auto arrayType = llvm::dyn_cast<ArrayType>(type);
    if (!arrayType) {
      return false;
    }
    return llvm::isa<FeltType>(arrayType.getElementType());
  }

  LogicalResult emitAmbiguousContainmentRhs(EmitContainmentOp containOp, StringRef detail) const {
    return containOp.emitOpError()
        << "poly lowering cannot resolve containment RHS row write history: "
        << detail;
  }

  /// Returns true when \p index begins with the exact attribute sequence in \p prefix.
  template <typename IndexRange, typename PrefixRange>
  bool indexStartsWith(const IndexRange &index, const PrefixRange &prefix) const {
    auto indexIt = index.begin();
    for (Attribute attr : prefix) {
      if (indexIt == index.end() || *indexIt != attr) {
        return false;
      }
      ++indexIt;
    }
    return true;
  }

  /// Returns true when \p lhs and \p rhs share a common prefix (one is a prefix of the other).
  template <typename LhsRange, typename RhsRange>
  bool prefixesCanOverlap(const LhsRange &lhs, const RhsRange &rhs) const {
    auto lhsIt = lhs.begin();
    auto rhsIt = rhs.begin();
    while (lhsIt != lhs.end() && rhsIt != rhs.end()) {
      if (*lhsIt != *rhsIt) {
        return false;
      }
      ++lhsIt;
      ++rhsIt;
    }
    return true;
  }

  /// Returns a new ArrayAttr with the first \p prefixSize elements of \p index removed.
  ArrayAttr dropIndexPrefix(MLIRContext *ctx, ArrayAttr index, size_t prefixSize) const {
    SmallVector<Attribute> attrs;
    size_t idx = 0;
    for (Attribute attr : index) {
      if (idx++ >= prefixSize) {
        attrs.push_back(attr);
      }
    }
    return ArrayAttr::get(ctx, attrs);
  }

  /// Returns a new ArrayAttr formed by concatenating \p prefix and \p suffix.
  template <typename PrefixRange>
  ArrayAttr appendIndex(MLIRContext *ctx, const PrefixRange &prefix, ArrayAttr suffix) const {
    SmallVector<Attribute> attrs;
    for (Attribute attr : prefix) {
      attrs.push_back(attr);
    }
    for (Attribute attr : suffix) {
      attrs.push_back(attr);
    }
    return ArrayAttr::get(ctx, attrs);
  }

  /// Returns the static access indices of \p op as an ArrayAttr.
  ArrayAttr getStaticAccessIndex(Operation *op) const {
    return llvm::cast<ArrayAccessOpInterface>(op).indexOperandsToAttributeArray();
  }

  /// Returns the subelement indices of \p arrayType that start with \p viewPrefix,
  /// with the prefix stripped from each result.
  std::optional<SmallVector<ArrayAttr>>
  getViewIndices(ArrayType arrayType, ArrayRef<Attribute> viewPrefix) const {
    std::optional<SmallVector<ArrayAttr>> allIndices = arrayType.getSubelementIndices();
    if (!allIndices) {
      return std::nullopt;
    }

    SmallVector<ArrayAttr> viewIndices;
    MLIRContext *ctx = arrayType.getContext();
    for (ArrayAttr index : *allIndices) {
      if (indexStartsWith(index, viewPrefix)) {
        viewIndices.push_back(dropIndexPrefix(ctx, index, viewPrefix.size()));
      }
    }
    return viewIndices;
  }

  /// Collects mutable felt-array element operands visible at \p boundaryOp,
  /// keyed by indices relative to \p viewPrefix.  Ambiguous write histories
  /// are rejected explicitly.
  LogicalResult collectMutableContainmentElements(
      Value arrayValue, Operation *boundaryOp, ArrayRef<Attribute> viewPrefix,
      EmitContainmentOp containOp, DenseSet<Value> &activeArrays,
      SmallVectorImpl<MutableContainmentElement> &elements
  ) {
    auto arrayType = llvm::dyn_cast<ArrayType>(arrayValue.getType());
    if (!arrayType || !llvm::isa<FeltType>(arrayType.getElementType())) {
      return success();
    }

    DenseMap<Attribute, OpOperand *> finalElements;
    if (failed(collectMutableContainmentElementMap(
            arrayValue, boundaryOp, viewPrefix, containOp, activeArrays, finalElements
        ))) {
      return failure();
    }

    std::optional<SmallVector<ArrayAttr>> viewIndices = getViewIndices(arrayType, viewPrefix);
    if (!viewIndices) {
      if (finalElements.empty()) {
        return success();
      }
      return emitAmbiguousContainmentRhs(containOp, "array shape is not static");
    }

    for (ArrayAttr relativeIndex : *viewIndices) {
      auto elementIt = finalElements.find(relativeIndex);
      if (elementIt != finalElements.end()) {
        elements.push_back(MutableContainmentElement {relativeIndex, elementIt->second});
      }
    }
    return success();
  }

  /// If \p v is defined by a `struct.readm`, returns the canonical
  /// (component, member) pair that identifies the source struct member.
  /// Returns std::nullopt when the defining op is not a struct read.
  static std::optional<std::pair<Value, FlatSymbolRefAttr>> resolveStructReadSource(Value v) {
    auto readOp = llvm::dyn_cast_if_present<MemberReadOp>(v.getDefiningOp());
    if (!readOp) {
      return std::nullopt;
    }
    return std::make_pair(readOp.getComponent(), readOp.getMemberNameAttr());
  }

  /// Returns true when \p a and \p b may reference the same underlying array
  /// data.  Two `struct.readm` values alias when they read the same member
  /// from the same component value (ignoring offset / map operands which are
  /// not expected in containment RHS paths).  For all other defining ops we
  /// fall back to strict Value equality.
  static bool mayAliasArraySource(Value a, Value b) {
    if (a == b) {
      return true;
    }
    auto srcA = resolveStructReadSource(a);
    if (!srcA) {
      return false;
    }
    auto srcB = resolveStructReadSource(b);
    if (!srcB) {
      return false;
    }
    return srcA->first == srcB->first && srcA->second == srcB->second;
  }

  /// Walks the write history of \p arrayValue up to \p boundaryOp and populates
  /// \p finalElements with the final visible felt operands, keyed by indices
  /// relative to \p viewPrefix.  Ambiguous histories are rejected.
  LogicalResult collectMutableContainmentElementMap(
      Value arrayValue, Operation *boundaryOp, ArrayRef<Attribute> viewPrefix,
      EmitContainmentOp containOp, DenseSet<Value> &activeArrays,
      DenseMap<Attribute, OpOperand *> &finalElements
  ) {
    // Track the final visible felt-array operands at boundaryOp, keyed by
    // indices relative to viewPrefix. Ambiguous histories fail instead of guessing.
    auto arrayType = llvm::dyn_cast<ArrayType>(arrayValue.getType());
    if (!arrayType || !llvm::isa<FeltType>(arrayType.getElementType())) {
      return success();
    }
    if (!boundaryOp || !boundaryOp->getBlock()) {
      return emitAmbiguousContainmentRhs(containOp, "missing observation block");
    }
    if (!activeArrays.insert(arrayValue).second) {
      return emitAmbiguousContainmentRhs(containOp, "cyclic array update");
    }
    auto cleanup = llvm::make_scope_exit([&]() { activeArrays.erase(arrayValue); });

    MLIRContext *ctx = arrayType.getContext();

    if (auto arrayOp = arrayValue.getDefiningOp<CreateArrayOp>()) {
      MutableOperandRange elementOperands = arrayOp.getElementsMutable();
      if (!elementOperands.empty()) {
        std::optional<SmallVector<ArrayAttr>> allIndices = arrayType.getSubelementIndices();
        if (!allIndices) {
          return emitAmbiguousContainmentRhs(containOp, "array.new shape is not static");
        }
        assert(allIndices->size() == elementOperands.size() && "array.new verifier mismatch");

        auto indexIt = allIndices->begin();
        for (OpOperand &elementOperand : elementOperands) {
          ArrayAttr index = *indexIt++;
          if (indexStartsWith(index, viewPrefix)) {
            finalElements[dropIndexPrefix(ctx, index, viewPrefix.size())] = &elementOperand;
          }
        }
      }
    }

    if (auto extractOp = arrayValue.getDefiningOp<ExtractArrayOp>()) {
      ArrayAttr extractIndex = getStaticAccessIndex(extractOp.getOperation());
      if (!extractIndex) {
        return emitAmbiguousContainmentRhs(containOp, "array.extract index is not static");
      }

      SmallVector<Attribute> sourcePrefix;
      for (Attribute attr : extractIndex) {
        sourcePrefix.push_back(attr);
      }
      for (Attribute attr : viewPrefix) {
        sourcePrefix.push_back(attr);
      }

      if (failed(collectMutableContainmentElementMap(
              extractOp.getArrRef(), extractOp.getOperation(), sourcePrefix, containOp,
              activeArrays, finalElements
          ))) {
        return failure();
      }
    }

    for (Operation &op : *boundaryOp->getBlock()) {
      if (&op == boundaryOp) {
        break;
      }

      if (auto writeOp = llvm::dyn_cast<WriteArrayOp>(&op)) {
        if (!mayAliasArraySource(writeOp.getArrRef(), arrayValue)) {
          continue;
        }

        ArrayAttr writeIndex = getStaticAccessIndex(writeOp.getOperation());
        if (!writeIndex) {
          return emitAmbiguousContainmentRhs(containOp, "array.write index is not static");
        }
        if (indexStartsWith(writeIndex, viewPrefix)) {
          finalElements[dropIndexPrefix(ctx, writeIndex, viewPrefix.size())] =
              &writeOp.getRvalueMutable();
        }
        continue;
      }

      if (auto insertOp = llvm::dyn_cast<InsertArrayOp>(&op)) {
        if (!mayAliasArraySource(insertOp.getArrRef(), arrayValue)) {
          continue;
        }

        ArrayAttr insertIndex = getStaticAccessIndex(insertOp.getOperation());
        if (!insertIndex) {
          return emitAmbiguousContainmentRhs(containOp, "array.insert index is not static");
        }
        if (!prefixesCanOverlap(insertIndex, viewPrefix)) {
          continue;
        }

        auto rvalueType = llvm::dyn_cast<ArrayType>(insertOp.getRvalue().getType());
        if (!rvalueType || !llvm::isa<FeltType>(rvalueType.getElementType())) {
          continue;
        }

        std::optional<SmallVector<ArrayAttr>> rvalueIndices = rvalueType.getSubelementIndices();
        if (!rvalueIndices) {
          return emitAmbiguousContainmentRhs(containOp, "array.insert rvalue shape is not static");
        }

        SmallVector<MutableContainmentElement> insertedElements;
        if (failed(collectMutableContainmentElements(
                insertOp.getRvalue(), insertOp.getOperation(), ArrayRef<Attribute> {}, containOp,
                activeArrays, insertedElements
            ))) {
          return failure();
        }

        DenseMap<Attribute, OpOperand *> insertedElementMap;
        for (MutableContainmentElement element : insertedElements) {
          insertedElementMap[element.index] = element.operand;
        }

        for (ArrayAttr rvalueIndex : *rvalueIndices) {
          ArrayAttr targetIndex = appendIndex(ctx, insertIndex, rvalueIndex);
          if (!indexStartsWith(targetIndex, viewPrefix)) {
            continue;
          }

          ArrayAttr relativeIndex = dropIndexPrefix(ctx, targetIndex, viewPrefix.size());
          auto elementIt = insertedElementMap.find(rvalueIndex);
          if (elementIt == insertedElementMap.end()) {
            finalElements.erase(relativeIndex);
            continue;
          }
          finalElements[relativeIndex] = elementIt->second;
        }
      }
    }

    return success();
  }

  /// Lowers an individual felt operand when its degree exceeds maxDegree,
  /// rewriting it through the existing auxiliary-member lowering path.
  LogicalResult lowerContainmentRhsFeltOperand(
      OpOperand &operand, StructDefOp structDef, FuncDefOp constrainFunc,
      DominanceInfo &dominanceInfo, DenseMap<Value, unsigned> &degreeMemo,
      DenseMap<Value, Value> &rewrites, SmallVector<AuxAssignment> &auxAssignments
  ) {
    Value value = operand.get();
    if (!llvm::isa<FeltType>(value.getType())) {
      return success();
    }

    unsigned degree = getDegree(value, degreeMemo);
    if (degree > maxDegree) {
      operand.set(lowerExpression(
          value, structDef, constrainFunc, operand.getOwner(), dominanceInfo, degreeMemo, rewrites,
          auxAssignments
      ));
    }
    return success();
  }

  /// Lowers the RHS operand of an EmitContainmentOp, recursing into
  /// felt-array elements when the RHS is an array type.
  LogicalResult lowerContainmentRhsValue(
      OpOperand &operand, StructDefOp structDef, FuncDefOp constrainFunc,
      DominanceInfo &dominanceInfo, DenseMap<Value, unsigned> &degreeMemo,
      DenseMap<Value, Value> &rewrites, SmallVector<AuxAssignment> &auxAssignments,
      EmitContainmentOp containOp
  ) {
    Value value = operand.get();
    if (llvm::isa<FeltType>(value.getType())) {
      return lowerContainmentRhsFeltOperand(
          operand, structDef, constrainFunc, dominanceInfo, degreeMemo, rewrites, auxAssignments
      );
    }

    if (!isFeltArray(value.getType())) {
      return success();
    }

    DenseSet<Value> activeArrays;
    SmallVector<MutableContainmentElement> elements;
    if (failed(collectMutableContainmentElements(
            value, containOp.getOperation(), ArrayRef<Attribute> {}, containOp, activeArrays,
            elements
        ))) {
      return failure();
    }

    for (MutableContainmentElement element : elements) {
      if (failed(lowerContainmentRhsFeltOperand(
              *element.operand, structDef, constrainFunc, dominanceInfo, degreeMemo, rewrites,
              auxAssignments
          ))) {
        return failure();
      }
    }

    return success();
  }

  /// Reports an error when a felt value in a containment RHS exceeds maxDegree
  /// after lowering has completed.
  void checkContainmentRhsFeltValue(
      Value value, EmitContainmentOp containOp, DenseMap<Value, unsigned> &checkMemo,
      bool &failedCheck
  ) {
    if (!llvm::isa<FeltType>(value.getType())) {
      return;
    }

    unsigned valueDegree = getDegree(value, checkMemo);
    if (valueDegree <= maxDegree) {
      return;
    }

    auto diag = containOp.emitOpError();
    diag << "poly lowering postcondition failed: containment RHS element degree "
            "exceeds max-degree "
         << maxDegree.getValue() << " (element degree " << valueDegree << ')';
    diag.report();
    failedCheck = true;
  }

  /// Recursively checks that every felt element visible in a containment RHS
  /// stays within maxDegree after lowering.
  LogicalResult checkContainmentRhsValue(
      Value value, EmitContainmentOp containOp, DenseMap<Value, unsigned> &checkMemo,
      bool &failedCheck
  ) {
    if (llvm::isa<FeltType>(value.getType())) {
      checkContainmentRhsFeltValue(value, containOp, checkMemo, failedCheck);
      return success();
    }

    if (!isFeltArray(value.getType())) {
      return success();
    }

    DenseSet<Value> activeArrays;
    SmallVector<MutableContainmentElement> elements;
    if (failed(collectMutableContainmentElements(
            value, containOp.getOperation(), ArrayRef<Attribute> {}, containOp, activeArrays,
            elements
        ))) {
      return failure();
    }

    for (MutableContainmentElement element : elements) {
      checkContainmentRhsFeltValue(element.operand->get(), containOp, checkMemo, failedCheck);
    }
    return success();
  }

  /// Postcondition: walks every EmitContainmentOp in \p constrainFunc and
  /// verifies that no containment RHS felt element exceeds maxDegree.
  LogicalResult checkContainmentRhsDegrees(FuncDefOp constrainFunc) {
    bool failedCheck = false;
    bool failedCollection = false;
    constrainFunc.walk([&](EmitContainmentOp containOp) {
      DenseMap<Value, unsigned> checkMemo;
      if (failed(checkContainmentRhsValue(containOp.getRhs(), containOp, checkMemo, failedCheck))) {
        failedCollection = true;
      }
    });
    return failure(failedCheck || failedCollection);
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Validate degree parameter
    if (maxDegree < 2) {
      auto diag = moduleOp.emitError();
      diag << "Invalid max degree: " << maxDegree.getValue() << ". Must be >= 2.";
      diag.report();
      signalPassFailure();
      return;
    }

    moduleOp.walk([this](StructDefOp structDef) {
      FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
      FuncDefOp computeFunc = structDef.getComputeFuncOp();
      if (!constrainFunc) {
        auto diag = structDef.emitOpError();
        diag << '"' << structDef.getName() << "\" doesn't have a \"@" << FUNC_NAME_CONSTRAIN
             << "\" function";
        diag.report();
        signalPassFailure();
        return;
      }

      if (!computeFunc) {
        auto diag = structDef.emitOpError();
        diag << '"' << structDef.getName() << "\" doesn't have a \"@" << FUNC_NAME_COMPUTE
             << "\" function";
        diag.report();
        signalPassFailure();
        return;
      }

      if (failed(checkForAuxMemberConflicts(structDef, AUXILIARY_MEMBER_PREFIX))) {
        signalPassFailure();
        return;
      }

      if (failed(checkFuncBodyIsStraightLine(constrainFunc, "poly lowering"))) {
        signalPassFailure();
        return;
      }

      if (failed(checkFuncBodyIsStraightLine(computeFunc, "poly lowering"))) {
        signalPassFailure();
        return;
      }

      DenseMap<Value, unsigned> degreeMemo;
      DenseMap<Value, Value> rewrites;
      SmallVector<AuxAssignment> auxAssignments;
      DominanceInfo dominanceInfo(constrainFunc);

      // Lower equality constraints
      constrainFunc.walk([&](EmitEqualityOp constraintOp) {
        auto &lhsOperand = constraintOp.getLhsMutable();
        auto &rhsOperand = constraintOp.getRhsMutable();
        unsigned degreeLhs = getDegree(lhsOperand.get(), degreeMemo);
        unsigned degreeRhs = getDegree(rhsOperand.get(), degreeMemo);

        if (degreeLhs > maxDegree) {
          Value loweredExpr = lowerExpression(
              lhsOperand.get(), structDef, constrainFunc, constraintOp.getOperation(),
              dominanceInfo, degreeMemo, rewrites, auxAssignments
          );
          lhsOperand.set(loweredExpr);
        }
        if (degreeRhs > maxDegree) {
          Value loweredExpr = lowerExpression(
              rhsOperand.get(), structDef, constrainFunc, constraintOp.getOperation(),
              dominanceInfo, degreeMemo, rewrites, auxAssignments
          );
          rhsOperand.set(loweredExpr);
        }
      });

      // Lower containment lookup rows.
      bool failedContainmentLowering = false;
      constrainFunc.walk([&](EmitContainmentOp containOp) {
        if (failed(lowerContainmentRhsValue(
                containOp.getRhsMutable(), structDef, constrainFunc, dominanceInfo, degreeMemo,
                rewrites, auxAssignments, containOp
            ))) {
          failedContainmentLowering = true;
        }
      });
      if (failedContainmentLowering) {
        signalPassFailure();
        return;
      }

      // Lower function call arguments
      constrainFunc.walk([&](CallOp callOp) {
        if (callOp.calleeIsStructConstrain()) {
          SmallVector<Value> newOperands = llvm::to_vector(callOp.getArgOperands());
          bool modified = false;

          for (Value &arg : newOperands) {
            if (!llvm::isa<FeltType>(arg.getType())) {
              continue;
            }

            DenseMap<Value, unsigned> callMemo;
            unsigned deg = getDegree(arg, callMemo);

            if (deg > 1) {
              arg = materializeCallArgument(
                  arg, structDef, constrainFunc, callOp, dominanceInfo, degreeMemo, rewrites,
                  auxAssignments
              );
              modified = true;
            }
          }

          if (modified) {
            OpBuilder builder(callOp);
            builder.create<CallOp>(
                callOp.getLoc(), callOp.getResultTypes(), callOp.getCallee(),
                CallOp::toVectorOfValueRange(callOp.getMapOperands()), callOp.getNumDimsPerMap(),
                newOperands
            );
            callOp->erase();
          }
        }
      });

      if (failed(checkEqualityDegrees(constrainFunc))) {
        signalPassFailure();
        return;
      }

      if (failed(checkContainmentRhsDegrees(constrainFunc))) {
        signalPassFailure();
        return;
      }

      if (failed(checkStructConstrainCallArguments(constrainFunc))) {
        signalPassFailure();
        return;
      }

      DenseMap<Value, Value> rebuildMemo;
      Block &computeBlock = computeFunc.getBody().front();
      OpBuilder builder(&computeBlock, computeBlock.getTerminator()->getIterator());
      Value selfVal = computeFunc.getSelfValueFromCompute();

      SmallVector<unsigned> orderedAuxAssignments;
      orderedAuxAssignments.reserve(auxAssignments.size());
      if (failed(orderAuxAssignments(auxAssignments, orderedAuxAssignments))) {
        signalPassFailure();
        return;
      }

      for (unsigned assignIdx : orderedAuxAssignments) {
        const auto &assign = auxAssignments[assignIdx];
        Value rebuiltExpr =
            rebuildExprInCompute(assign.computedValue, computeFunc, builder, rebuildMemo);
        if (!rebuiltExpr) {
          signalPassFailure();
          return;
        }
        builder.create<MemberWriteOp>(
            assign.computedValue.getLoc(), selfVal, builder.getStringAttr(assign.auxMemberName),
            rebuiltExpr
        );
        if (assign.auxValue) {
          // Reuse the expression just written so later aux producers do not need an
          // immediate read from the generated aux member.
          rebuildMemo[assign.auxValue] = rebuiltExpr;
        }
      }
    });
  }
};

} // namespace
