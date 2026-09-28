//===-- R1CSLoweringPass.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared R1CS normalization and the preparation, legacy, and direct passes.
///
//===----------------------------------------------------------------------===//

#include "r1cs/Dialect/IR/Attrs.h"
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"
#include "r1cs/Target/R1CSBinary.h"
#include "r1cs/Transforms/TransformationPasses.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/Transforms/ConstraintEvaluation.h"
#include "llzk/Transforms/LoweringUtils.h"
#include "llzk/Util/Constants.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/Walk.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Matchers.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace r1cs {
#define GEN_PASS_DEF_R1CSLOWERINGPASS
#define GEN_PASS_DEF_R1CSPREPAREPASS
#define GEN_PASS_DEF_R1CSDIRECTLOWERINGPASS
#include "r1cs/Transforms/TransformationPasses.h.inc"
} // namespace r1cs

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

#define DEBUG_TYPE "llzk-r1cs-lowering"
#define R1CS_AUXILIARY_MEMBER_PREFIX "__llzk_r1cs_lowering_pass_aux_member_"

namespace {

/// A LinearCombination is a map from a Value (like a variable or MemberRead) to a felt constant.
struct LinearCombination {
  DenseMap<Value, DynamicAPInt> terms; // variable -> coeff
  DynamicAPInt constant;

  LinearCombination() : constant() {}

  void addTerm(Value v, const DynamicAPInt &coeff) {
    if (coeff == 0) {
      return;
    }
    auto [it, inserted] = terms.try_emplace(v, coeff);
    if (!inserted) {
      it->second += coeff;
    }
  }

  void addTerm(Value v, int64_t coeff) {
    DynamicAPInt dynamicCoeff(coeff);
    return addTerm(v, dynamicCoeff);
  }

  void negate() {
    for (auto &kv : terms) {
      kv.second = -kv.second;
    }
    constant = -constant;
  };

  LinearCombination scaled(const DynamicAPInt &factor) const {
    LinearCombination result;
    if (factor == 0) {
      return result;
    }

    for (const auto &kv : terms) {
      result.terms[kv.first] = kv.second * factor;
    }
    result.constant = constant * factor;
    return result;
  }

  LinearCombination scaled(int64_t factor) const {
    DynamicAPInt dynamicFactor(factor);
    return scaled(dynamicFactor);
  }

  LinearCombination add(const LinearCombination &other) const {
    LinearCombination result(*this);

    for (const auto &kv : other.terms) {
      auto [it, inserted] = result.terms.try_emplace(kv.first, kv.second);
      if (!inserted) {
        it->second += kv.second;
      }
    }
    result.constant += other.constant;
    return result;
  }

  LinearCombination negated() const { return scaled(-1); }

  void print(raw_ostream &os) const {
    bool first = true;
    for (const auto &[val, coeff] : terms) {
      if (!first) {
        os << " + ";
      }
      first = false;
      os << coeff << '*' << val;
    }
    if (constant != 0) {
      if (!first) {
        os << " + ";
      }
      os << constant;
    }
    if (first && constant == 0) {
      os << '0';
    }
  }
};

/// A struct representing a * b = c R1CS constraint
struct R1CSConstraint {
  LinearCombination a;
  LinearCombination b;
  LinearCombination c;

  R1CSConstraint negated() const {
    R1CSConstraint result(*this);
    result.a = a.negated();
    result.c = c.negated();
    return result;
  }

  R1CSConstraint scaled(const DynamicAPInt &factor) const {
    R1CSConstraint result(*this);
    result.a = a.scaled(factor);
    result.c = c.scaled(factor);
    return result;
  }

  R1CSConstraint(const DynamicAPInt &constant) { c.constant = constant; }

  R1CSConstraint() = default;

  inline bool isLinearOnly() const { return a.terms.empty() && b.terms.empty(); }

  R1CSConstraint multiply(const R1CSConstraint &other) {
    auto isDegZero = [](const R1CSConstraint &constraint) {
      return constraint.a.terms.empty() && constraint.b.terms.empty() && constraint.c.terms.empty();
    };

    if (isDegZero(other)) {
      return this->scaled(other.c.constant);
    }
    if (isDegZero(*this)) {
      return other.scaled(this->c.constant);
    }

    if (isLinearOnly() && other.isLinearOnly()) {
      R1CSConstraint result;
      result.a = this->c;
      result.b = other.c;

      // We do NOT compute `c = a * b` because R1CS doesn't need it explicitly
      // It suffices to enforce: a * b = c

      return result;
    }
    llvm::errs() << "R1CSConstraint::multiply: Only supported for purely linear constraints.\n";
    llvm_unreachable("Invalid multiply: non-linear constraint(s) involved");
  }

  R1CSConstraint add(const R1CSConstraint &other) {

    if (isLinearOnly()) {
      R1CSConstraint result(other);
      result.c = result.c.add(this->c);
      return result;
    }
    if (other.isLinearOnly()) {
      R1CSConstraint result(*this);
      result.c = result.c.add(other.c);
      return result;
    }
    llvm::errs() << "R1CSConstraint::add: Only supported for purely linear constraints.\n";
    llvm_unreachable("Invalid add: non-linear constraint(s) involved");
  }

  void print(raw_ostream &os) const {
    os << '(';
    a.print(os);
    os << ") * (";
    b.print(os);
    os << ") = ";
    c.print(os);
  }
};

/// Shared normalization and circuit construction, independent of pass dispatch.
class R1CSLowering {
  unsigned auxCounter = 0;

public:
  // Normalize a felt-valued expression into R1CS-compatible form.
  // This performs *minimal* rewriting:
  // - Only rewrites Add/Sub of two degree-2 terms
  // - Operates bottom-up using post-order traversal
  //
  // Resulting expression is R1CS-compatible (i.e., one multiplication per constraint)
  // and can be directly used in EmitEqualityOp or as operands of other expressions.
  static void getPostOrder(Value root, SmallVectorImpl<Value> &postOrder) {
    SmallVector<Value, 16> worklist;
    DenseSet<Value> visited;

    worklist.push_back(root);

    while (!worklist.empty()) {
      Value val = worklist.back();

      if (!visited.insert(val).second) {
        worklist.pop_back();
        postOrder.push_back(val);
        continue;
      }

      if (Operation *op = val.getDefiningOp()) {
        if (isa<MemberReadOp, array::ReadArrayOp, pod::ReadPodOp>(op)) {
          continue;
        }
        for (Value operand : op->getOperands()) {
          worklist.push_back(operand);
        }
      }
    }
  }

  /// Normalize a felt-valued expression into R1CS-compatible form by rewriting
  /// only when strictly necessary. This function ensures the resulting expression:
  ///
  /// - Has at most one multiplication per constraint (R1CS-compatible)
  /// - Avoids unnecessary introduction of auxiliary variables
  /// - Preserves semantic equivalence via auxiliary member equality constraints
  ///
  /// Rewriting is done **bottom-up** using post-order traversal of the def-use chain.
  /// The transformation is minimal:
  /// - Only rewrites Add/Sub where both operands are degree-2
  /// - Leaves multiplications intact unless their operands require rewriting due to constants
  /// - Avoids rewriting expressions that are already linear or already normalized
  ///
  /// The function memoizes all degrees and rewrites for efficiency and correctness,
  /// and records any auxiliary member assignments for later reconstruction in compute().
  ///
  /// \param root           The root felt-valued expression to normalize.
  /// \param structDef      The enclosing struct definition (for adding aux members).
  /// \param constrainFunc  The constrain() function containing the constraint logic.
  /// \param degreeMemo     Memoized degrees of expressions (to avoid recomputation).
  /// \param rewrites       Memoized rewrites of expressions.
  /// \param auxAssignments Records auxiliary member assignments introduced during normalization.
  /// \param builder        Builder used to insert new ops in the constrain() block.
  /// \returns              A Value representing the normalized (possibly rewritten) expression.
  FailureOr<Value> normalizeForR1CS(
      Value root, StructDefOp structDef, FuncDefOp constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo, DenseMap<Value, Value> &rewrites,
      SmallVectorImpl<AuxAssignment> &auxAssignments, OpBuilder &builder
  ) {
    if (auto it = rewrites.find(root); it != rewrites.end()) {
      return it->second;
    }

    // Use an insertion guard to restore the builder's insertion point after this function.
    // This avoids invalid references that could occur in the caller after executing this
    // function since `handleAddOrSub` sets the insertion point to an op that it may erase.
    OpBuilder::InsertionGuard guard(builder);

    SmallVector<Value, 16> postOrder;
    getPostOrder(root, postOrder);

    // We perform a bottom up rewrite of the expressions. For any expression e := op(e_1, ...,
    // e_n) we first rewrite e_1, ..., e_n if necessary and then rewrite e based on op.
    for (Value val : postOrder) {
      if (rewrites.contains(val)) {
        continue;
      }

      Operation *op = val.getDefiningOp();

      if (!op) {
        // Block arguments, etc.
        degreeMemo[val] = 1;
        rewrites[val] = val;
        continue;
      }

      // Case 1: Felt constant op. The degree is 0 and no rewrite is needed.
      if (auto c = llvm::dyn_cast<FeltConstantOp>(op)) {
        degreeMemo[val] = 0;
        rewrites[val] = val;
        continue;
      }

      // Case 2: Member read op. The degree is 1 and no rewrite needed.
      if (isa<MemberReadOp, array::ReadArrayOp, pod::ReadPodOp>(op)) {
        degreeMemo[val] = 1;
        rewrites[val] = val;
        continue;
      }

      // Helper function for getting degree from memo map
      auto getDeg = [&degreeMemo](Value v) -> unsigned {
        auto it = degreeMemo.find(v);
        assert(it != degreeMemo.end() && "Missing degree");
        return it->second;
      };

      // Case 3: lhs +/- rhs. There are three subcases cases to consider:
      // 1) If deg(lhs) <= degree(rhs) < 2 then nothing needs to be done
      // 2) If deg(lhs) = 2 and degree(rhs) < 2 then nothing further has to be done.
      // 3) If deg(lhs) = deg(rhs) = 2 then we lower one of lhs or rhs.
      auto handleAddOrSub = [this, &rewrites, &getDeg, &builder, op, structDef, &val,
                             &constrainFunc, &auxAssignments,
                             &degreeMemo](Value lhsOrig, Value rhsOrig, bool isAdd) {
        Value lhs = rewrites[lhsOrig];
        Value rhs = rewrites[rhsOrig];
        unsigned degLhs = getDeg(lhs);
        unsigned degRhs = getDeg(rhs);

        if (degLhs == 2 && degRhs == 2) {
          builder.setInsertionPoint(op);
          std::string auxName = R1CS_AUXILIARY_MEMBER_PREFIX + std::to_string(auxCounter++);
          MemberDefOp auxMember = addAuxMember(structDef, auxName, val.getType());
          Value aux = MemberReadOp::create(
              builder, val.getLoc(), val.getType(), constrainFunc.getSelfValueFromConstrain(),
              auxMember.getNameAttr()
          );
          auto eqOp = EmitEqualityOp::create(builder, val.getLoc(), aux, lhs);
          auxAssignments.push_back({auxName, lhs});
          degreeMemo[aux] = 1;
          rewrites[aux] = aux;
          replaceSubsequentUsesWith(lhs, aux, eqOp);
          lhs = aux;
          degLhs = 1;

          Operation *newOp =
              isAdd ? AddFeltOp::create(builder, val.getLoc(), val.getType(), lhs, rhs)
                    : SubFeltOp::create(builder, val.getLoc(), val.getType(), lhs, rhs);
          Value result = newOp->getResult(0);
          degreeMemo[result] = std::max(degLhs, degRhs);
          rewrites[val] = result;
          rewrites[result] = result;
          val.replaceAllUsesWith(result);
          if (val.use_empty()) {
            op->erase();
          }
        } else {
          degreeMemo[val] = std::max(degLhs, degRhs);
          rewrites[val] = val;
        }
      };

      if (auto add = llvm::dyn_cast<AddFeltOp>(op)) {
        handleAddOrSub(add.getLhs(), add.getRhs(), /*isAdd=*/true);
        continue;
      }

      if (auto sub = llvm::dyn_cast<SubFeltOp>(op)) {
        handleAddOrSub(sub.getLhs(), sub.getRhs(), /*isAdd=*/false);
        continue;
      }

      // Case 4: lhs * rhs. Nothing further needs to be done assuming the degree lowering pass has
      // been run with maxDegree = 2. This is because both operands are normalized and at most one
      // operand can be quadratic.
      if (auto mul = llvm::dyn_cast<MulFeltOp>(op)) {
        Value lhs = rewrites[mul.getLhs()];
        Value rhs = rewrites[mul.getRhs()];
        unsigned degLhs = getDeg(lhs);
        unsigned degRhs = getDeg(rhs);

        degreeMemo[val] = degLhs + degRhs;
        rewrites[val] = val;
        continue;
      }

      // Case 6: Neg. Similar to multiplication, nothing needs to be done since we are doing the
      // rewrite bottom up
      if (auto neg = llvm::dyn_cast<NegFeltOp>(op)) {
        Value inner = rewrites[neg.getOperand()];
        unsigned deg = getDeg(inner);
        degreeMemo[val] = deg;
        rewrites[val] = val;
        continue;
      }

      return op->emitError("unsupported operation in R1CS normalization");
    }

    return rewrites[root];
  }

  static FailureOr<R1CSConstraint> lowerPolyToR1CS(Value poly) {
    DenseMap<Value, R1CSConstraint> constraintMap;
    SmallVector<Value, 16> postorder;
    getPostOrder(poly, postorder);

    // Bottom-up construction of R1CSConstraints
    for (Value v : postorder) {
      Operation *op = v.getDefiningOp();
      if (!op || isa<MemberReadOp, array::ReadArrayOp, pod::ReadPodOp>(op)) {
        // Leaf (input variable or member read)
        R1CSConstraint eq;
        eq.c.addTerm(v, 1);
        constraintMap[v] = eq;
        continue;
      }
      if (auto add = llvm::dyn_cast<AddFeltOp>(op)) {
        R1CSConstraint lhsC = constraintMap[add.getLhs()];
        R1CSConstraint rhsC = constraintMap[add.getRhs()];
        constraintMap[v] = lhsC.add(rhsC);
      } else if (auto sub = llvm::dyn_cast<SubFeltOp>(op)) {
        R1CSConstraint lhsC = constraintMap[sub.getLhs()];
        R1CSConstraint rhsC = constraintMap[sub.getRhs()];
        constraintMap[v] = lhsC.add(rhsC.negated());
      } else if (auto mul = llvm::dyn_cast<MulFeltOp>(op)) {
        R1CSConstraint lhsC = constraintMap[mul.getLhs()];
        R1CSConstraint rhsC = constraintMap[mul.getRhs()];
        constraintMap[v] = lhsC.multiply(rhsC);
      } else if (auto neg = llvm::dyn_cast<NegFeltOp>(op)) {
        R1CSConstraint inner = constraintMap[op->getOperand(0)];
        constraintMap[v] = inner.negated();
      } else if (auto cst = llvm::dyn_cast<FeltConstantOp>(op)) {
        R1CSConstraint c(toDynamicAPInt(cst.getValue()));
        constraintMap[v] = c;
      } else {
        return op->emitError("unsupported operation in R1CS lowering");
      }
    }

    return constraintMap[poly];
  }

  static FailureOr<R1CSConstraint>
  lowerEquationToR1CS(Value p, Value q, const DenseMap<Value, unsigned> &degreeMemo) {
    auto lhs = lowerPolyToR1CS(p);
    if (failed(lhs)) {
      return failure();
    }
    auto rhs = lowerPolyToR1CS(q);
    if (failed(rhs)) {
      return failure();
    }
    R1CSConstraint &pconst = *lhs, &qconst = *rhs;

    if (degreeMemo.at(p) == 2) {
      if (degreeMemo.at(q) == 2) {
        return emitError(p.getLoc(), "R1CS lowering requires at most one quadratic side");
      }
      R1CSConstraint result(pconst);
      result.c = qconst.c.add(pconst.c.negated());
      return result;
    }
    if (degreeMemo.at(q) == 2) {
      R1CSConstraint result(qconst);
      result.c = pconst.c.add(qconst.c.negated());
      return result;
    }
    return qconst.add(pconst.negated());
  }

  FailureOr<Value> emitLinearCombination(
      const LinearCombination &lc, IRMapping &valueMap, DenseMap<StringRef, Value> &memberMap,
      Value selfVal, OpBuilder &builder, Location loc
  ) {
    Value result = nullptr;

    auto getMapping = [&valueMap, &memberMap, selfVal](const Value &v) -> FailureOr<Value> {
      if (!valueMap.contains(v)) {
        Operation *op = v.getDefiningOp();
        if (auto read = llvm::dyn_cast<MemberReadOp>(op)) {
          if (read.getComponent() != selfVal) {
            return read.emitError(
                "R1CS lowering only supports member reads rooted at the current constrain "
                "self value"
            );
          }
          // Table offsets and map operands select a different row.  Those
          // accesses must not be lowered to the current-row R1CS signal.
          if (read.getTableOffset() || !read.getMapOperands().empty()) {
            return read.emitError(
                "R1CS lowering does not support member reads with table offsets "
                "or map operands"
            );
          }
          auto memberVal = memberMap.find(read.getMemberName());
          if (memberVal == memberMap.end()) {
            return read.emitError("member read is not associated with an R1CS signal");
          }
          return memberVal->second;
        }
        return op->emitError("Value not mapped in R1CS lowering");
      }
      return valueMap.lookup(v);
    };

    auto linearTy = r1cs::LinearType::get(builder.getContext());

    // Start with the constant, if present
    if (lc.constant != 0) {
      result = r1cs::ConstOp::create(
          builder, loc, linearTy, r1cs::FeltAttr::get(builder.getContext(), toAPSInt(lc.constant))
      );
    }

    for (const auto &[val, coeff] : lc.terms) {
      FailureOr<Value> mapped = getMapping(val);
      if (failed(mapped)) {
        return failure();
      }
      // %tmp = r1cs.to_linear %mapped
      // most of these will be removed with CSE passes
      Value lin = r1cs::ToLinearOp::create(builder, loc, linearTy, *mapped);
      // %scaled = r1cs.mul_const %lin, coeff
      Value scaled = coeff == 1 ? lin
                                : r1cs::MulConstOp::create(
                                      builder, loc, linearTy, lin,
                                      r1cs::FeltAttr::get(builder.getContext(), toAPSInt(coeff))
                                  );

      // Accumulate via r1cs.add
      if (!result) {
        result = scaled;
      } else {
        result = r1cs::AddOp::create(builder, loc, linearTy, result, scaled);
      }
    }

    if (!result) {
      // Entire linear combination was zero
      result = r1cs::ConstOp::create(
          builder, loc, r1cs::LinearType::get(builder.getContext()),
          r1cs::FeltAttr::get(builder.getContext(), toAPSInt(lc.constant))
      );
    }

    return result;
  }

  /// Lower direct storage reads without changing the rolled witness layout.
  /// The serialized binding list follows the binary exporter's physical wire order.
  LogicalResult buildEvaluatedR1CS(
      ModuleOp module, StructDefOp def, FuncDefOp function, DenseMap<Value, unsigned> &degrees
  ) {
    OpBuilder top(module.getBodyRegion());
    struct Signal {
      ArrayAttr path;
      bool isPublic;
      bool input;
      SmallVector<Value> values;
    };
    SmallVector<Signal> signals;
    DenseMap<Attribute, unsigned> positions;
    auto add = [&positions, &signals](ArrayAttr path, bool pub, Value value = {}) {
      auto [it, inserted] = positions.try_emplace(path, signals.size());
      if (inserted) {
        signals.push_back({path, pub, cast<IntegerAttr>(path[0]).getInt() != 0, {}});
      }
      assert(signals[it->second].isPublic == pub && "inconsistent visibility for storage path");
      if (value) {
        signals[it->second].values.push_back(value);
      }
    };
    // Keep complete scalar/array input and public-output interfaces, including
    // unconstrained leaves, rather than inferring visibility from used reads.
    std::function<LogicalResult(Type, SmallVector<Attribute>, bool)> addLeaves;
    addLeaves = [&add, &top, &def,
                 &addLeaves](Type type, SmallVector<Attribute> path, bool pub) -> LogicalResult {
      if (isa<FeltType>(type)) {
        add(top.getArrayAttr(path), pub);
        return success();
      }
      if (auto array = dyn_cast<array::ArrayType>(type)) {
        if (llvm::any_of(array.getShape(), [](int64_t size) { return size < 0; })) {
          return def.emitError("R1CS interface requires statically shaped arrays");
        }
        std::function<LogicalResult(unsigned)> dimension =
            [&addLeaves, array, &path, pub, &dimension, &top](unsigned d) -> LogicalResult {
          if (d == array.getRank()) {
            return addLeaves(array.getElementType(), path, pub);
          }
          for (int64_t i = 0; i < array.getShape()[d]; ++i) {
            path.push_back(top.getIndexAttr(i));
            if (failed(dimension(d + 1))) {
              return failure();
            }
            path.pop_back();
          }
          return success();
        };
        return dimension(0);
      }
      return def.emitError("R1CS interface requires felt values or static felt arrays");
    };
    for (auto arg : llvm::drop_begin(function.getArguments())) {
      if (failed(addLeaves(
              arg.getType(), {top.getI64IntegerAttr(arg.getArgNumber())},
              function.hasArgPublicAttr(arg.getArgNumber())
          ))) {
        return failure();
      }
      if (llvm::isa<FeltType>(arg.getType())) {
        add(top.getArrayAttr({top.getI64IntegerAttr(arg.getArgNumber())}),
            function.hasArgPublicAttr(arg.getArgNumber()), arg);
      }
    }
    for (auto member : def.getMemberDefs()) {
      auto original = member->getAttrOfType<BoolAttr>(polymorphic::ORIGINAL_PUBLIC_ATTR_NAME);
      bool pub = original ? original.getValue() : member.hasPublicAttr();
      if (pub || llvm::isa<FeltType>(member.getType())) {
        if (failed(
                addLeaves(member.getType(), {top.getI64IntegerAttr(0), member.getNameAttr()}, pub)
            )) {
          return failure();
        }
      }
    }
    // Metadata describes storage; it must agree with the actual SSA access chain.
    SymbolTableCollection tables;
    DenseMap<Value, polymorphic::StorageBinding> storage;
    std::function<FailureOr<polymorphic::StorageBinding>(Value)> resolveStorage;
    resolveStorage = [&storage, &function, &top, &resolveStorage,
                      &tables](Value value) -> FailureOr<polymorphic::StorageBinding> {
      if (auto found = storage.find(value); found != storage.end()) {
        return found->second;
      }
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        if (argument.getOwner() != &function.getBody().front()) {
          return failure();
        }
        unsigned number = argument.getArgNumber();
        return storage[value] = {
                   top.getArrayAttr({top.getI64IntegerAttr(number)}),
                   number == 0 || function.hasArgPublicAttr(number)
               };
      }
      Operation *read = value.getDefiningOp();
      if (!read || !isa<MemberReadOp, array::ReadArrayOp, pod::ReadPodOp>(read)) {
        return failure();
      }
      auto parent = resolveStorage(read->getOperand(0));
      if (failed(parent)) {
        return failure();
      }
      SmallVector<Attribute> path(parent->path.getValue());
      bool pub = parent->isPublic;
      if (auto memberRead = dyn_cast<MemberReadOp>(read)) {
        if (memberRead.getTableOffset() || !memberRead.getMapOperands().empty()) {
          return failure();
        }
        auto member = memberRead.getMemberDefOp(tables);
        if (failed(member)) {
          return failure();
        }
        auto original =
            member->get()->getAttrOfType<BoolAttr>(polymorphic::ORIGINAL_PUBLIC_ATTR_NAME);
        pub &= original ? original.getValue() : member->get().hasPublicAttr();
        path.push_back(top.getStringAttr(memberRead.getMemberName()));
      } else if (auto arrayRead = dyn_cast<array::ReadArrayOp>(read)) {
        auto type = cast<array::ArrayType>(read->getOperand(0).getType());
        if (arrayRead.getIndices().size() != static_cast<size_t>(type.getRank())) {
          return failure();
        }
        for (auto [index, size] : llvm::zip(arrayRead.getIndices(), type.getShape())) {
          APInt constant;
          if (!matchPattern(index, m_ConstantInt(&constant)) || constant.isNegative() ||
              constant.getActiveBits() > 63 || size < 0 ||
              constant.getZExtValue() >= static_cast<uint64_t>(size)) {
            return failure();
          }
          path.push_back(top.getIndexAttr(constant.getZExtValue()));
        }
      } else {
        path.push_back(top.getStringAttr(cast<pod::ReadPodOp>(read).getRecordName()));
      }
      return storage[value] = {top.getArrayAttr(path), pub};
    };
    auto reads =
        function.walk([&function, &resolveStorage, &add, &def, &top](Operation *op) -> WalkResult {
      if (!isa<MemberReadOp, array::ReadArrayOp, pod::ReadPodOp>(op) ||
          !isa<FeltType>(op->getResult(0).getType())) {
        return WalkResult::advance();
      }
      if (auto attr = op->getAttr(polymorphic::SIGNAL_BINDING_ATTR_NAME)) {
        auto binding = polymorphic::getStorageBinding(attr);
        if (failed(binding) ||
            cast<IntegerAttr>(binding->path[0]).getInt() >= function.getNumArguments()) {
          return op->emitError("invalid evaluated signal storage binding");
        }
        auto actual = resolveStorage(op->getResult(0));
        auto sameSegment = [](Attribute lhs, Attribute rhs) {
          if (auto left = dyn_cast<IntegerAttr>(lhs)) {
            auto right = dyn_cast<IntegerAttr>(rhs);
            return right && left.getInt() == right.getInt();
          }
          return lhs == rhs;
        };
        if (failed(actual) || actual->isPublic != binding->isPublic ||
            actual->path.size() != binding->path.size() ||
            !llvm::all_of(llvm::zip(actual->path, binding->path), [sameSegment](auto pair) {
          return sameSegment(std::get<0>(pair), std::get<1>(pair));
        })) {
          return op->emitError("signal binding does not match the storage read");
        }
        add(actual->path, actual->isPublic, op->getResult(0));
      } else if (auto read = llvm::dyn_cast<MemberReadOp>(op);
                 read && read.getComponent() == function.getArgument(0) && !read.getTableOffset() &&
                 read.getMapOperands().empty()) {
        auto member = def.getMemberDef(top.getStringAttr(read.getMemberName()));
        auto original = member->getAttrOfType<BoolAttr>(polymorphic::ORIGINAL_PUBLIC_ATTR_NAME);
        add(top.getArrayAttr({top.getI64IntegerAttr(0), top.getStringAttr(read.getMemberName())}),
            original ? original.getValue() : member.hasPublicAttr(), read.getResult());
      } else {
        return op->emitError("direct R1CS storage read has no signal binding");
      }
      return WalkResult::advance();
    });
    if (reads.wasInterrupted()) {
      return failure();
    }
    NamedAttrList inputAttrs;
    unsigned inputCount = 0;
    for (auto &signal : signals) {
      if (signal.input && signal.isPublic) {
        inputAttrs.set(std::to_string(inputCount), top.getAttr<r1cs::PublicAttr>());
      }
      inputCount += signal.input;
    }
    auto circuit = r1cs::CircuitDefOp::create(
        top, def.getLoc(), (def.getSymName() + "__r1cs").str(),
        inputAttrs.getDictionary(top.getContext())
    );
    OpBuilder body = OpBuilder::atBlockEnd(circuit.addEntryBlock());
    IRMapping values;
    DenseMap<StringRef, Value> unused;
    uint32_t label = 1;
    for (auto &signal : signals) {
      Value wire;
      if (signal.input) {
        wire =
            circuit.getBody().front().addArgument(body.getType<r1cs::SignalType>(), def.getLoc());
      } else {
        wire = r1cs::SignalDefOp::create(
                   body, def.getLoc(), body.getType<r1cs::SignalType>(),
                   body.getUI32IntegerAttr(label++),
                   signal.isPublic ? body.getAttr<r1cs::PublicAttr>() : r1cs::PublicAttr()
        )
                   .getOut();
      }
      for (auto value : signal.values) {
        values.map(value, wire);
      }
    }
    SmallVector<Attribute> wireBindings;
    for (auto [input, pub] :
         {std::pair {false, true}, {true, true}, {true, false}, {false, false}}) {
      for (auto &signal : signals) {
        if (signal.input != input || signal.isPublic != pub) {
          continue;
        }
        NamedAttrList binding;
        binding.set("path", signal.path);
        binding.set("public", body.getBoolAttr(pub));
        binding.set("wire", body.getI64IntegerAttr(wireBindings.size() + 1));
        wireBindings.push_back(binding.getDictionary(body.getContext()));
      }
    }
    circuit->setAttr(r1cs::WIRE_BINDINGS_ATTR_NAME, body.getArrayAttr(wireBindings));
    module->setAttr(r1cs::CIRCUIT_REF_ATTR_NAME, SymbolRefAttr::get(circuit));
    for (auto eq : function.getBody().front().getOps<EmitEqualityOp>()) {
      getFeltDegree(eq.getLhs(), degrees);
      getFeltDegree(eq.getRhs(), degrees);
      auto constraint = lowerEquationToR1CS(eq.getLhs(), eq.getRhs(), degrees);
      if (failed(constraint)) {
        return failure();
      }
      auto a = emitLinearCombination(
          constraint->a, values, unused, function.getArgument(0), body, eq.getLoc()
      );
      auto b = emitLinearCombination(
          constraint->b, values, unused, function.getArgument(0), body, eq.getLoc()
      );
      auto c = emitLinearCombination(
          constraint->c, values, unused, function.getArgument(0), body, eq.getLoc()
      );
      if (failed(a) || failed(b) || failed(c)) {
        return failure();
      }
      r1cs::ConstrainOp::create(body, eq.getLoc(), *a, *b, *c);
    }
    return success();
  }

  LogicalResult buildAndEmitR1CS(
      ModuleOp &moduleOp, StructDefOp &structDef, FuncDefOp &constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo
  ) {
    // Validate struct members are felt and prepare signal types for circuit result types
    bool hasPublicSignals = false;
    for (auto member : structDef.getMemberDefs()) {
      if (!llvm::isa<FeltType>(member.getType())) {
        return member.emitError("Only felt members are supported as output signals");
      }
      if (member.isPublic()) {
        hasPublicSignals = true;
      }
    }
    if (!hasPublicSignals) {
      structDef.emitWarning("Struct should have at least one public output").report();
    }

    Region &constrainFuncBody = constrainFunc.getBody();

    SmallVector<R1CSConstraint> constraints;
    auto lowered =
        constrainFuncBody.walk([&degreeMemo, &constraints](EmitEqualityOp eqOp) -> WalkResult {
      // A prepare-only invocation may have normalized the IR in another process.
      getFeltDegree(eqOp.getLhs(), degreeMemo);
      getFeltDegree(eqOp.getRhs(), degreeMemo);
      auto constraint = lowerEquationToR1CS(eqOp.getLhs(), eqOp.getRhs(), degreeMemo);
      if (failed(constraint)) {
        return WalkResult::interrupt();
      }
      constraints.push_back(*constraint);
      return WalkResult::advance();
    });
    if (lowered.wasInterrupted()) {
      return failure();
    }

    OpBuilder topBuilder(moduleOp.getBodyRegion());
    moduleOp->setAttr(LANG_ATTR_NAME, topBuilder.getStringAttr("r1cs"));

    IRMapping valueMap;
    Location loc = structDef.getLoc();
    SmallVector<mlir::NamedAttribute> argAttrPairs;
    auto inputArgs = llvm::enumerate(llvm::drop_begin(constrainFuncBody.front().getArguments()));
    for (auto [i, arg] : inputArgs) {
      if (constrainFunc.hasArgPublicAttr(i + 1)) {
        auto key = topBuilder.getStringAttr(std::to_string(i));
        auto value = r1cs::PublicAttr::get(moduleOp.getContext());
        argAttrPairs.emplace_back(key, value);
      }
    }
    auto circuit = r1cs::CircuitDefOp::create(
        topBuilder, loc, structDef.getSymName().str(), topBuilder.getDictionaryAttr(argAttrPairs)
    );

    Block *circuitBlock = circuit.addEntryBlock();

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(circuitBlock);

    // Step 3: Validate that all parameters to the constrain function are felt types
    for (auto [i, arg] : inputArgs) {
      if (!llvm::isa<FeltType>(arg.getType())) {
        return constrainFunc.emitOpError("All input arguments must be of felt type");
      }
      auto blockArg = circuitBlock->addArgument(bodyBuilder.getType<r1cs::SignalType>(), loc);
      valueMap.map(arg, blockArg);
    }

    // Step 4: For every struct member we a) create a signaldefop and b) add that signal to our
    // outputs
    DenseMap<StringRef, Value> memberSignalMap;
    // Label 0 belongs to the implicit constant-one wire in the binary R1CS
    // format. Keep source signal labels compatible with binary export.
    uint32_t signalDefCntr = 1;
    for (auto member : structDef.getMemberDefs()) {
      r1cs::PublicAttr pubAttr;
      if (member.hasPublicAttr()) {
        pubAttr = bodyBuilder.getAttr<r1cs::PublicAttr>();
      }
      auto defOp = r1cs::SignalDefOp::create(
          bodyBuilder, member.getLoc(), bodyBuilder.getType<r1cs::SignalType>(),
          bodyBuilder.getUI32IntegerAttr(signalDefCntr), pubAttr
      );
      signalDefCntr++;
      memberSignalMap.insert({member.getName(), defOp.getOut()});
    }

    // Step 5: Emit the R1CS constraints
    Value selfVal = constrainFunc.getSelfValueFromConstrain();
    for (const R1CSConstraint &constraint : constraints) {
      FailureOr<Value> aVal =
          emitLinearCombination(constraint.a, valueMap, memberSignalMap, selfVal, bodyBuilder, loc);
      if (failed(aVal)) {
        return failure();
      }
      FailureOr<Value> bVal =
          emitLinearCombination(constraint.b, valueMap, memberSignalMap, selfVal, bodyBuilder, loc);
      if (failed(bVal)) {
        return failure();
      }
      FailureOr<Value> cVal =
          emitLinearCombination(constraint.c, valueMap, memberSignalMap, selfVal, bodyBuilder, loc);
      if (failed(cVal)) {
        return failure();
      }
      r1cs::ConstrainOp::create(bodyBuilder, loc, *aVal, *bVal, *cVal);
    }
    return success();
  }

  /// Normalize one struct and install all auxiliary compute assignments once.
  LogicalResult prepare(StructDefOp structDef) {
    FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
    FuncDefOp computeFunc = structDef.getComputeFuncOp();
    if (!constrainFunc || !computeFunc) {
      structDef.emitOpError("Missing compute or constrain function").report();
      return failure();
    }

    if (!structDef->hasAttr(r1cs::PREPARED_ATTR_NAME) &&
        failed(checkForAuxMemberConflicts(structDef, R1CS_AUXILIARY_MEMBER_PREFIX))) {
      return failure();
    }

    if (failed(checkFuncBodyIsStraightLine(constrainFunc, "R1CS lowering"))) {
      return failure();
    }

    DenseMap<Value, unsigned> degreeMemo;
    DenseMap<Value, Value> rewrites;
    SmallVector<AuxAssignment> auxAssignments;

    if (!structDef->hasAttr(r1cs::PREPARED_ATTR_NAME)) {
      auto normalized = constrainFunc.walk(
          [this, structDef, &constrainFunc, &degreeMemo, &rewrites,
           &auxAssignments](EmitEqualityOp eqOp) -> WalkResult {
        OpBuilder builder(eqOp);
        auto lhsResult = normalizeForR1CS(
            eqOp.getLhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments, builder
        );
        if (failed(lhsResult)) {
          return WalkResult::interrupt();
        }
        auto rhsResult = normalizeForR1CS(
            eqOp.getRhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments, builder
        );

        if (failed(rhsResult)) {
          return WalkResult::interrupt();
        }
        Value lhs = *lhsResult, rhs = *rhsResult;
        unsigned degLhs = degreeMemo.lookup(lhs);
        unsigned degRhs = degreeMemo.lookup(rhs);

        // If both sides are degree 2, isolate one side
        if (degLhs == 2 && degRhs == 2) {
          std::string auxName = R1CS_AUXILIARY_MEMBER_PREFIX + std::to_string(auxCounter++);
          MemberDefOp auxMember = addAuxMember(structDef, auxName, lhs.getType());
          Value aux = MemberReadOp::create(
              builder, eqOp.getLoc(), lhs.getType(), constrainFunc.getSelfValueFromConstrain(),
              auxMember.getNameAttr()
          );
          auto eqAux = EmitEqualityOp::create(builder, eqOp.getLoc(), aux, lhs);
          auxAssignments.push_back({auxName, lhs});
          degreeMemo[aux] = 1;
          replaceSubsequentUsesWith(lhs, aux, eqAux);
          lhs = aux;
        }

        EmitEqualityOp::create(builder, eqOp.getLoc(), lhs, rhs);
        eqOp.erase();
        return WalkResult::advance();
      }
      );
      if (normalized.wasInterrupted()) {
        return failure();
      }

      if (!auxAssignments.empty() && computeFunc.isExternal()) {
        computeFunc.emitError("R1CS auxiliaries require a compute body");
        return failure();
      }
      SmallVector<Value> expressions;
      for (const auto &assign : auxAssignments) {
        expressions.push_back(assign.computedValue);
      }
      DenseMap<Value, Value> captured;
      if (failed(captureAuxiliaryInputs(expressions, computeFunc, captured))) {
        return failure();
      }
      for (Block &computeBlock : computeFunc.getBody()) {
        auto ret = dyn_cast<ReturnOp>(computeBlock.getTerminator());
        if (!ret) {
          continue;
        }
        OpBuilder builder(ret);
        Value selfVal = ret.getOperand(0);
        DenseMap<Value, Value> rebuildMemo = captured;
        rebuildMemo[constrainFunc.getArgument(0)] = selfVal;
        for (const auto &assign : auxAssignments) {
          Value expr =
              rebuildExprInCompute(assign.computedValue, computeFunc, builder, rebuildMemo);
          if (!expr) {
            return failure();
          }
          MemberWriteOp::create(
              builder, assign.computedValue.getLoc(), selfVal,
              builder.getStringAttr(assign.auxMemberName), expr
          );
        }
      }
      structDef->setAttr(r1cs::PREPARED_ATTR_NAME, UnitAttr::get(structDef.getContext()));
    }
    return success();
  }
};

/// Normalize constraints without choosing an emission strategy.
class R1CSPreparePass : public r1cs::impl::R1CSPreparePassBase<R1CSPreparePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    bool evaluated = polymorphic::isEvaluatedModule(module);
    R1CSLowering lowering;
    bool changed = false;
    auto result = module.walk([&lowering, evaluated, &changed](StructDefOp def) -> WalkResult {
      if ((evaluated && !def.isMainComponent()) || def->hasAttr(r1cs::PREPARED_ATTR_NAME)) {
        return WalkResult::skip();
      }
      if (failed(lowering.prepare(def))) {
        return WalkResult::interrupt();
      }
      changed = true;
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    } else if (!changed) {
      markAllAnalysesPreserved();
    }
  }
};

/// Preserve the legacy flattened-input API, rejecting evaluated storage layouts.
class R1CSLoweringPass : public r1cs::impl::R1CSLoweringPassBase<R1CSLoweringPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<r1cs::R1CSDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (polymorphic::isEvaluatedModule(module)) {
      module.emitError(
          "llzk-r1cs-lowering rejects poly.evaluated_main; use llzk-r1cs-direct-lowering"
      );
      signalPassFailure();
      return;
    }
    R1CSLowering lowering;
    auto result = module.walk([&lowering, module](StructDefOp def) mutable -> WalkResult {
      if (failed(lowering.prepare(def))) {
        return WalkResult::interrupt();
      }
      DenseMap<Value, unsigned> degrees;
      auto constrain = def.getConstrainFuncOp();
      if (failed(lowering.buildAndEmitR1CS(module, def, constrain, degrees))) {
        return WalkResult::interrupt();
      }
      def.erase();
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    eraseEmptyNestedModules(module);
    module->removeAttr(MAIN_ATTR_NAME);
  }
};

/// Emit only the evaluated main circuit while retaining its witness storage.
class R1CSDirectLoweringPass
    : public r1cs::impl::R1CSDirectLoweringPassBase<R1CSDirectLoweringPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<r1cs::R1CSDialect>();
  }
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!polymorphic::isEvaluatedModule(module)) {
      module.emitError(
          "llzk-r1cs-direct-lowering requires poly.evaluated_main; run llzk-evaluate-constraints "
          "first"
      );
      signalPassFailure();
      return;
    }
    if (module->hasAttr(r1cs::CIRCUIT_REF_ATTR_NAME)) {
      markAllAnalysesPreserved();
      return;
    }
    SymbolTableCollection tables;
    auto main = getMainInstanceDef(tables, module);
    if (failed(main) || !*main) {
      module.emitError("direct R1CS lowering requires the original evaluated main struct");
      signalPassFailure();
      return;
    }
    auto def = main->get();
    auto evaluatedMain =
        module->getAttrOfType<SymbolRefAttr>(polymorphic::EVALUATED_MAIN_ATTR_NAME);
    if (!evaluatedMain || evaluatedMain != def.getFullyQualifiedName()) {
      module.emitError(
          "direct R1CS lowering requires the original evaluated main struct; do not pre-flatten "
          "evaluated input"
      );
      signalPassFailure();
      return;
    }
    R1CSLowering lowering;
    if (failed(lowering.prepare(def))) {
      signalPassFailure();
      return;
    }
    DenseMap<Value, unsigned> degrees;
    if (failed(lowering.buildEvaluatedR1CS(module, def, def.getConstrainFuncOp(), degrees))) {
      signalPassFailure();
    }
  }
};

} // namespace
