//===-- LLZKR1CSLoweringPass.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-r1cs-lowering` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "r1cs/Dialect/IR/Attrs.h"
#include "r1cs/Dialect/IR/Ops.h"
#include "r1cs/Dialect/IR/Types.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DEF_R1CSLOWERINGPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

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

    if (!terms.contains(v)) {
      terms[v] = coeff;
    } else {
      terms[v] += coeff;
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
      if (!result.terms.contains(kv.first)) {
        result.terms[kv.first] = kv.second;
      } else {
        result.terms[kv.first] = result.terms[kv.first] + kv.second;
      }
    }
    result.constant = result.constant + other.constant;
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

class R1CSLoweringPass : public llzk::impl::R1CSLoweringPassBase<R1CSLoweringPass> {
private:
  unsigned auxCounter = 0;

  // Normalize a felt-valued expression into R1CS-compatible form.
  // This performs *minimal* rewriting:
  // - Only rewrites Add/Sub of two degree-2 terms
  // - Operates bottom-up using post-order traversal
  //
  // Resulting expression is R1CS-compatible (i.e., one multiplication per constraint)
  // and can be directly used in EmitEqualityOp or as operands of other expressions.

  void getPostOrder(Value root, SmallVectorImpl<Value> &postOrder) {
    SmallVector<Value, 16> worklist;
    DenseSet<Value> visited;

    worklist.push_back(root);

    while (!worklist.empty()) {
      Value val = worklist.back();

      if (visited.contains(val)) {
        worklist.pop_back();
        postOrder.push_back(val);
        continue;
      }

      visited.insert(val);
      if (Operation *op = val.getDefiningOp()) {
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
  Value normalizeForR1CS(
      Value root, StructDefOp structDef, FuncDefOp constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo, DenseMap<Value, Value> &rewrites,
      SmallVectorImpl<AuxAssignment> &auxAssignments, OpBuilder &builder
  ) {
    if (auto it = rewrites.find(root); it != rewrites.end()) {
      return it->second;
    }

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
      if (auto fr = llvm::dyn_cast<MemberReadOp>(op)) {
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
      auto handleAddOrSub = [&](Value lhsOrig, Value rhsOrig, bool isAdd) {
        Value lhs = rewrites[lhsOrig];
        Value rhs = rewrites[rhsOrig];
        unsigned degLhs = getDeg(lhs);
        unsigned degRhs = getDeg(rhs);

        if (degLhs == 2 && degRhs == 2) {
          builder.setInsertionPoint(op);
          std::string auxName = R1CS_AUXILIARY_MEMBER_PREFIX + std::to_string(auxCounter++);
          MemberDefOp auxMember = addAuxMember(structDef, auxName);
          Value aux = builder.create<MemberReadOp>(
              val.getLoc(), val.getType(), constrainFunc.getSelfValueFromConstrain(),
              auxMember.getNameAttr()
          );
          auto eqOp = builder.create<EmitEqualityOp>(val.getLoc(), aux, lhs);
          auxAssignments.push_back({auxName, lhs});
          degreeMemo[aux] = 1;
          rewrites[aux] = aux;
          replaceSubsequentUsesWith(lhs, aux, eqOp);
          lhs = aux;
          degLhs = 1;

          Operation *newOp = isAdd
                                 ? builder.create<AddFeltOp>(val.getLoc(), val.getType(), lhs, rhs)
                                 : builder.create<SubFeltOp>(val.getLoc(), val.getType(), lhs, rhs);
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

      llvm::errs() << "Unhandled op in normalize ForR1CS: " << *op << '\n';
      signalPassFailure();
    }

    return rewrites[root];
  }

  R1CSConstraint lowerPolyToR1CS(Value poly) {
    // Worklist-based post-order traversal
    SmallVector<Value, 16> worklist = {poly};
    DenseMap<Value, R1CSConstraint> constraintMap;
    DenseSet<Value> visited;
    SmallVector<Value, 16> postorder;

    getPostOrder(poly, postorder);

    // Bottom-up construction of R1CSConstraints
    for (Value v : postorder) {
      Operation *op = v.getDefiningOp();
      if (!op || llvm::isa<MemberReadOp>(op)) {
        // Leaf (input variable or member read)
        R1CSConstraint eq;
        eq.c.addTerm(v, 1);
        constraintMap[v] = eq;
        continue;
      }
      if (auto add = dyn_cast<AddFeltOp>(op)) {
        R1CSConstraint lhsC = constraintMap[add.getLhs()];
        R1CSConstraint rhsC = constraintMap[add.getRhs()];
        constraintMap[v] = lhsC.add(rhsC);
      } else if (auto sub = dyn_cast<SubFeltOp>(op)) {
        R1CSConstraint lhsC = constraintMap[sub.getLhs()];
        R1CSConstraint rhsC = constraintMap[sub.getRhs()];
        constraintMap[v] = lhsC.add(rhsC.negated());
      } else if (auto mul = dyn_cast<MulFeltOp>(op)) {
        R1CSConstraint lhsC = constraintMap[mul.getLhs()];
        R1CSConstraint rhsC = constraintMap[mul.getRhs()];
        constraintMap[v] = lhsC.multiply(rhsC);
      } else if (auto neg = dyn_cast<NegFeltOp>(op)) {
        R1CSConstraint inner = constraintMap[op->getOperand(0)];
        constraintMap[v] = inner.negated();
      } else if (auto cst = dyn_cast<FeltConstantOp>(op)) {
        R1CSConstraint c(toDynamicAPInt(cst.getValue()));
        constraintMap[v] = c;
      } else {
        llvm::errs() << "Unhandled op in R1CS lowering: " << *op << '\n';
        llvm_unreachable("unhandled op");
      }
    }

    return constraintMap[poly];
  }

  R1CSConstraint
  lowerEquationToR1CS(Value p, Value q, const DenseMap<Value, unsigned> &degreeMemo) {
    R1CSConstraint pconst = lowerPolyToR1CS(p);
    R1CSConstraint qconst = lowerPolyToR1CS(q);

    if (degreeMemo.at(p) == 2) {
      return pconst.add(qconst.negated());
    }
    return qconst.add(pconst.negated());
  }

  Value emitLinearCombination(
      LinearCombination lc, IRMapping &valueMap, DenseMap<StringRef, Value> &memberMap,
      OpBuilder &builder, Location loc
  ) {
    Value result = nullptr;

    auto getMapping = [&valueMap, &memberMap, this](const Value &v) {
      if (!valueMap.contains(v)) {
        Operation *op = v.getDefiningOp();
        if (auto read = dyn_cast<MemberReadOp>(op)) {
          auto memberVal = memberMap.find(read.getMemberName());
          assert(memberVal != memberMap.end() && "Member read not associated with a value");
          return memberVal->second;
        }
        op->emitError("Value not mapped in R1CS lowering").report();
        signalPassFailure();
      }
      return valueMap.lookup(v);
    };

    auto linearTy = r1cs::LinearType::get(builder.getContext());

    // Start with the constant, if present
    if (lc.constant != 0) {
      result = builder.create<r1cs::ConstOp>(
          loc, linearTy, r1cs::FeltAttr::get(builder.getContext(), toAPSInt(lc.constant))
      );
    }

    for (auto &[val, coeff] : lc.terms) {
      Value mapped = getMapping(val);
      // %tmp = r1cs.to_linear %mapped
      // most of these will be removed with CSE passes
      Value lin = builder.create<r1cs::ToLinearOp>(loc, linearTy, mapped);
      // %scaled = r1cs.mul_const %lin, coeff
      Value scaled = coeff == 1 ? lin
                                : builder.create<r1cs::MulConstOp>(
                                      loc, linearTy, lin,
                                      r1cs::FeltAttr::get(builder.getContext(), toAPSInt(coeff))
                                  );

      // Accumulate via r1cs.add
      if (!result) {
        result = scaled;
      } else {
        result = builder.create<r1cs::AddOp>(loc, linearTy, result, scaled);
      }
    }

    if (!result) {
      // Entire linear combination was zero
      result = builder.create<r1cs::ConstOp>(
          loc, r1cs::LinearType::get(builder.getContext()),
          r1cs::FeltAttr::get(builder.getContext(), 0)
      );
    }

    return result;
  }

  void buildAndEmitR1CS(
      ModuleOp &moduleOp, StructDefOp &structDef, FuncDefOp &constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo
  ) {
    SmallVector<R1CSConstraint, 16> constraints;
    constrainFunc.walk([&](EmitEqualityOp eqOp) {
      OpBuilder builder(eqOp);
      R1CSConstraint eq = lowerEquationToR1CS(eqOp.getLhs(), eqOp.getRhs(), degreeMemo);
      constraints.push_back(eq);
    });
    moduleOp->setAttr(LANG_ATTR_NAME, StringAttr::get(moduleOp.getContext(), "r1cs"));
    Block &entryBlock = constrainFunc.getBody().front();
    IRMapping valueMap;
    Location loc = structDef.getLoc();
    OpBuilder topBuilder(moduleOp.getBodyRegion());

    // Validate struct members are felt and prepare signal types for circuit result types
    bool hasPublicSignals = false;
    for (auto member : structDef.getMemberDefs()) {
      if (!llvm::isa<FeltType>(member.getType())) {
        member.emitError("Only felt members are supported as output signals").report();
        signalPassFailure();
        return;
      }
      if (member.isPublic()) {
        hasPublicSignals = true;
      }
    }

    if (!hasPublicSignals) {
      structDef.emitError("Struct should have at least one public output").report();
    }
    llvm::SmallVector<mlir::NamedAttribute> argAttrPairs;

    for (auto [i, arg] : llvm::enumerate(llvm::drop_begin(entryBlock.getArguments(), 1))) {
      if (constrainFunc.hasArgPublicAttr(i + 1)) {
        auto key = topBuilder.getStringAttr(std::to_string(i));
        auto value = r1cs::PublicAttr::get(moduleOp.getContext());
        argAttrPairs.emplace_back(key, value);
      }
    }
    auto dictAttr = topBuilder.getDictionaryAttr(argAttrPairs);
    auto circuit =
        topBuilder.create<r1cs::CircuitDefOp>(loc, structDef.getSymName().str(), dictAttr);

    Block *circuitBlock = circuit.addEntryBlock();

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(circuitBlock);

    // Step 3: Validate that all parameters to the constrain function are felt types
    for (auto [i, arg] : llvm::enumerate(llvm::drop_begin(entryBlock.getArguments(), 1))) {
      if (!llvm::isa<FeltType>(arg.getType())) {
        constrainFunc.emitOpError("All input arguments must be of felt type").report();
        signalPassFailure();
        return;
      }
      auto blockArg = circuitBlock->addArgument(bodyBuilder.getType<r1cs::SignalType>(), loc);
      valueMap.map(arg, blockArg);
    }

    // Step 4: For every struct member we a) create a signaldefop and b) add that signal to our
    // outputs
    DenseMap<StringRef, Value> memberSignalMap;
    uint32_t signalDefCntr = 0;
    for (auto member : structDef.getMemberDefs()) {
      r1cs::PublicAttr pubAttr;
      if (member.hasPublicAttr()) {
        pubAttr = bodyBuilder.getAttr<r1cs::PublicAttr>();
      }
      auto defOp = bodyBuilder.create<r1cs::SignalDefOp>(
          member.getLoc(), bodyBuilder.getType<r1cs::SignalType>(),
          bodyBuilder.getUI32IntegerAttr(signalDefCntr), pubAttr
      );
      signalDefCntr++;
      memberSignalMap.insert({member.getName(), defOp.getOut()});
    }
    DenseMap<std::tuple<Value, Value, StringRef>, Value> binaryOpCache;
    // Step 5: Emit the R1CS constraints
    for (const R1CSConstraint &constraint : constraints) {
      Value aVal = emitLinearCombination(constraint.a, valueMap, memberSignalMap, bodyBuilder, loc);
      Value bVal = emitLinearCombination(constraint.b, valueMap, memberSignalMap, bodyBuilder, loc);
      Value cVal = emitLinearCombination(constraint.c, valueMap, memberSignalMap, bodyBuilder, loc);
      bodyBuilder.create<r1cs::ConstrainOp>(loc, aVal, bVal, cVal);
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<r1cs::R1CSDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    assert(
        moduleOp->getContext()->getLoadedDialect<r1cs::R1CSDialect>() && "R1CS dialect not loaded"
    );
    moduleOp.walk([this, &moduleOp](StructDefOp structDef) {
      FuncDefOp constrainFunc = structDef.getConstrainFuncOp();
      FuncDefOp computeFunc = structDef.getComputeFuncOp();
      if (!constrainFunc || !computeFunc) {
        structDef.emitOpError("Missing compute or constrain function").report();
        signalPassFailure();
        return;
      }

      if (failed(checkForAuxMemberConflicts(structDef, R1CS_AUXILIARY_MEMBER_PREFIX))) {
        signalPassFailure();
        return;
      }

      DenseMap<Value, unsigned> degreeMemo;
      DenseMap<Value, Value> rewrites;
      SmallVector<AuxAssignment> auxAssignments;

      constrainFunc.walk([&](EmitEqualityOp eqOp) {
        OpBuilder builder(eqOp);
        Value lhs = normalizeForR1CS(
            eqOp.getLhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments, builder
        );
        Value rhs = normalizeForR1CS(
            eqOp.getRhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments, builder
        );

        unsigned degLhs = degreeMemo.lookup(lhs);
        unsigned degRhs = degreeMemo.lookup(rhs);

        // If both sides are degree 2, isolate one side
        if (degLhs == 2 && degRhs == 2) {
          builder.setInsertionPoint(eqOp);
          std::string auxName = R1CS_AUXILIARY_MEMBER_PREFIX + std::to_string(auxCounter++);
          MemberDefOp auxMember = addAuxMember(structDef, auxName);
          Value aux = builder.create<MemberReadOp>(
              eqOp.getLoc(), lhs.getType(), constrainFunc.getSelfValueFromConstrain(),
              auxMember.getNameAttr()
          );
          auto eqAux = builder.create<EmitEqualityOp>(eqOp.getLoc(), aux, lhs);
          auxAssignments.push_back({auxName, lhs});
          degreeMemo[aux] = 1;
          replaceSubsequentUsesWith(lhs, aux, eqAux);
          lhs = aux;
        }

        builder.create<EmitEqualityOp>(eqOp.getLoc(), lhs, rhs);
        eqOp.erase();
      });

      Block &computeBlock = computeFunc.getBody().front();
      OpBuilder builder(&computeBlock, computeBlock.getTerminator()->getIterator());
      Value selfVal = computeFunc.getSelfValueFromCompute();
      DenseMap<Value, Value> rebuildMemo;

      for (const auto &assign : auxAssignments) {
        Value expr = rebuildExprInCompute(assign.computedValue, computeFunc, builder, rebuildMemo);
        builder.create<MemberWriteOp>(
            assign.computedValue.getLoc(), selfVal, builder.getStringAttr(assign.auxMemberName),
            expr
        );
      }
      buildAndEmitR1CS(moduleOp, structDef, constrainFunc, degreeMemo);
      structDef.erase();
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createR1CSLoweringPass() {
  return std::make_unique<R1CSLoweringPass>();
}
