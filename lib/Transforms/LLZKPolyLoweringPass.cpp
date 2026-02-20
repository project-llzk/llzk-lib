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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

#include <deque>
#include <memory>

// Include the generated base pass class definitions.
namespace llzk {
#define GEN_PASS_DECL_POLYLOWERINGPASS
#define GEN_PASS_DEF_POLYLOWERINGPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

#define DEBUG_TYPE "llzk-poly-lowering-pass"
#define AUXILIARY_MEMBER_PREFIX "__llzk_poly_lowering_pass_aux_member_"

namespace {

struct AuxAssignment {
  std::string auxMemberName;
  Value computedValue;
};

class PolyLoweringPass : public llzk::impl::PolyLoweringPassBase<PolyLoweringPass> {
public:
  void setMaxDegree(unsigned degree) { this->maxDegree = degree; }

private:
  unsigned auxCounter = 0;

  void collectStructDefs(ModuleOp modOp, SmallVectorImpl<StructDefOp> &structDefs) {
    modOp.walk([&structDefs](StructDefOp structDef) {
      structDefs.push_back(structDef);
      return WalkResult::skip();
    });
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
      Value val, StructDefOp structDef, FuncDefOp constrainFunc,
      DenseMap<Value, unsigned> &degreeMemo, DenseMap<Value, Value> &rewrites,
      SmallVector<AuxAssignment> &auxAssignments
  ) {
    if (rewrites.count(val)) {
      return rewrites[val];
    }

    unsigned degree = getDegree(val, degreeMemo);
    if (degree <= maxDegree) {
      rewrites[val] = val;
      return val;
    }

    if (auto mulOp = val.getDefiningOp<MulFeltOp>()) {
      // Recursively lower operands first
      Value lhs = lowerExpression(
          mulOp.getLhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
      );
      Value rhs = lowerExpression(
          mulOp.getRhs(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
      );

      unsigned lhsDeg = getDegree(lhs, degreeMemo);
      unsigned rhsDeg = getDegree(rhs, degreeMemo);

      OpBuilder builder(mulOp.getOperation()->getBlock(), ++Block::iterator(mulOp));
      Value selfVal = constrainFunc.getSelfValueFromConstrain();
      bool eraseMul = lhsDeg + rhsDeg > maxDegree;
      // Optimization: If lhs == rhs, factor it only once
      if (lhs == rhs && eraseMul) {
        std::string auxName = AUXILIARY_MEMBER_PREFIX + std::to_string(this->auxCounter++);
        MemberDefOp auxMember = addAuxMember(structDef, auxName);

        auto auxVal = builder.create<MemberReadOp>(
            lhs.getLoc(), lhs.getType(), selfVal, auxMember.getNameAttr()
        );
        auxAssignments.push_back({auxName, lhs});
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
        MemberDefOp auxMember = addAuxMember(structDef, auxName);

        // Read back as MemberReadOp (new SSA value)
        auto auxVal = builder.create<MemberReadOp>(
            toFactor.getLoc(), toFactor.getType(), selfVal, auxMember.getNameAttr()
        );

        // Emit constraint: auxVal == toFactor
        Location loc = builder.getFusedLoc({auxVal.getLoc(), toFactor.getLoc()});
        auto eqOp = builder.create<EmitEqualityOp>(loc, auxVal, toFactor);
        auxAssignments.push_back({auxName, toFactor});
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

    // For non-mul ops, leave untouched (they're degree-1 safe)
    rewrites[val] = val;
    return val;
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

    moduleOp.walk([this, &moduleOp](StructDefOp structDef) {
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

      DenseMap<Value, unsigned> degreeMemo;
      DenseMap<Value, Value> rewrites;
      SmallVector<AuxAssignment> auxAssignments;

      // Lower equality constraints
      constrainFunc.walk([&](EmitEqualityOp constraintOp) {
        auto &lhsOperand = constraintOp.getLhsMutable();
        auto &rhsOperand = constraintOp.getRhsMutable();
        unsigned degreeLhs = getDegree(lhsOperand.get(), degreeMemo);
        unsigned degreeRhs = getDegree(rhsOperand.get(), degreeMemo);

        if (degreeLhs > maxDegree) {
          Value loweredExpr = lowerExpression(
              lhsOperand.get(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
          );
          lhsOperand.set(loweredExpr);
        }
        if (degreeRhs > maxDegree) {
          Value loweredExpr = lowerExpression(
              rhsOperand.get(), structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
          );
          rhsOperand.set(loweredExpr);
        }
      });

      // The pass doesn't currently support EmitContainmentOp.
      // See https://github.com/project-llzk/llzk-lib/issues/261
      constrainFunc.walk([this, &moduleOp](EmitContainmentOp /*containOp*/) {
        auto diag = moduleOp.emitError();
        diag << "EmitContainmentOp is unsupported for now in the lowering pass";
        diag.report();
        signalPassFailure();
        return;
      });

      // Lower function call arguments
      constrainFunc.walk([&](CallOp callOp) {
        if (callOp.calleeIsStructConstrain()) {
          SmallVector<Value> newOperands = llvm::to_vector(callOp.getArgOperands());
          bool modified = false;

          for (Value &arg : newOperands) {
            unsigned deg = getDegree(arg, degreeMemo);

            if (deg > 1) {
              Value loweredArg = lowerExpression(
                  arg, structDef, constrainFunc, degreeMemo, rewrites, auxAssignments
              );
              arg = loweredArg;
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

      DenseMap<Value, Value> rebuildMemo;
      Block &computeBlock = computeFunc.getBody().front();
      OpBuilder builder(&computeBlock, computeBlock.getTerminator()->getIterator());
      Value selfVal = computeFunc.getSelfValueFromCompute();

      for (const auto &assign : auxAssignments) {
        Value rebuiltExpr =
            rebuildExprInCompute(assign.computedValue, computeFunc, builder, rebuildMemo);
        builder.create<MemberWriteOp>(
            assign.computedValue.getLoc(), selfVal, builder.getStringAttr(assign.auxMemberName),
            rebuiltExpr
        );
      }
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> llzk::createPolyLoweringPass() {
  return std::make_unique<PolyLoweringPass>();
};

std::unique_ptr<mlir::Pass> llzk::createPolyLoweringPass(unsigned maxDegree) {
  auto pass = std::make_unique<PolyLoweringPass>();
  static_cast<PolyLoweringPass *>(pass.get())->setMaxDegree(maxDegree);
  return pass;
}
