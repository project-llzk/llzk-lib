//===-- LLZKLoweringUtils.cpp -----------------------------------*- C++ -*-===//
//
// Shared utility function implementations for LLZK lowering passes.
//
//===----------------------------------------------------------------------===//

#include "llzk/Transforms/LLZKLoweringUtils.h"

#include "llzk/Dialect/LLZK/IR/Ops.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::function;
using namespace llzk::component;
using namespace llzk::constrain;

namespace llzk {

namespace {

Value mapBlockArgumentInCompute(BlockArgument barg, FuncDefOp computeFunc) {
  // Constrain entry arguments map onto compute inputs: constrain(%self, args...)
  // corresponds to compute(args...), plus the compute-side `%self`.
  if (barg.getArgNumber() == 0) {
    return computeFunc.getSelfValueFromCompute();
  }
  return computeFunc.getArgument(barg.getArgNumber() - 1);
}

Value mapValueIntoCompute(
    Value val, FuncDefOp computeFunc, OpBuilder &builder, DenseMap<Value, Value> &memo
) {
  if (auto it = memo.find(val); it != memo.end()) {
    return it->second;
  }
  if (auto barg = llvm::dyn_cast<BlockArgument>(val)) {
    return memo[val] = mapBlockArgumentInCompute(barg, computeFunc);
  }
  return rebuildExprInCompute(val, computeFunc, builder, memo);
}

} // namespace

Value rebuildExprInCompute(
    Value val, FuncDefOp computeFunc, OpBuilder &builder, DenseMap<Value, Value> &memo
) {
  if (auto it = memo.find(val); it != memo.end()) {
    return it->second;
  }

  if (auto barg = llvm::dyn_cast<BlockArgument>(val)) {
    return memo[val] = mapBlockArgumentInCompute(barg, computeFunc);
  }

  if (auto readOp = val.getDefiningOp<MemberReadOp>()) {
    IRMapping mapper;
    for (Value operand : readOp->getOperands()) {
      mapper.map(operand, mapValueIntoCompute(operand, computeFunc, builder, memo));
    }

    Operation *rebuiltOp = builder.clone(*readOp.getOperation(), mapper);
    assert(rebuiltOp->getNumResults() == 1 && "member reads have exactly one result");
    return memo[val] = rebuiltOp->getResult(0);
  }

  if (val.getType().isIndex()) {
    // Preserve index producers used by member-read access operands so rebuilt reads
    // keep the original access semantics.
    Operation *defOp = val.getDefiningOp();
    assert(defOp && "index block arguments should already be mapped");

    IRMapping mapper;
    for (Value operand : defOp->getOperands()) {
      mapper.map(operand, mapValueIntoCompute(operand, computeFunc, builder, memo));
    }

    Operation *rebuiltOp = builder.clone(*defOp, mapper);
    assert(
        rebuiltOp->getNumResults() == defOp->getNumResults() &&
        "cloned index op should preserve result count"
    );
    unsigned resultNumber = llvm::cast<OpResult>(val).getResultNumber();
    return memo[val] = rebuiltOp->getResult(resultNumber);
  }

  if (auto add = val.getDefiningOp<AddFeltOp>()) {
    Value lhs = rebuildExprInCompute(add.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(add.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<AddFeltOp>(add.getLoc(), add.getType(), lhs, rhs);
  }

  if (auto sub = val.getDefiningOp<SubFeltOp>()) {
    Value lhs = rebuildExprInCompute(sub.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(sub.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<SubFeltOp>(sub.getLoc(), sub.getType(), lhs, rhs);
  }

  if (auto mul = val.getDefiningOp<MulFeltOp>()) {
    Value lhs = rebuildExprInCompute(mul.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(mul.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<MulFeltOp>(mul.getLoc(), mul.getType(), lhs, rhs);
  }

  if (auto neg = val.getDefiningOp<NegFeltOp>()) {
    Value operand = rebuildExprInCompute(neg.getOperand(), computeFunc, builder, memo);
    return memo[val] = builder.create<NegFeltOp>(neg.getLoc(), neg.getType(), operand);
  }

  if (auto div = val.getDefiningOp<DivFeltOp>()) {
    Value lhs = rebuildExprInCompute(div.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(div.getRhs(), computeFunc, builder, memo);
    return memo[val] = builder.create<DivFeltOp>(div.getLoc(), div.getType(), lhs, rhs);
  }

  if (auto c = val.getDefiningOp<FeltConstantOp>()) {
    return memo[val] = builder.create<FeltConstantOp>(c.getLoc(), c.getValueAttr());
  }

  llvm::errs() << "Unhandled op in rebuildExprInCompute: " << val << '\n';
  llvm_unreachable("Unsupported op kind");
}

LogicalResult checkForAuxMemberConflicts(StructDefOp structDef, StringRef prefix) {
  bool conflictFound = false;

  structDef.walk([&conflictFound, &prefix](MemberDefOp memberDefOp) {
    if (memberDefOp.getName().starts_with(prefix)) {
      (memberDefOp.emitError() << "Member name '" << memberDefOp.getName()
                               << "' conflicts with reserved prefix '" << prefix << '\'')
          .report();
      conflictFound = true;
    }
  });

  return failure(conflictFound);
}

LogicalResult checkFuncBodyIsStraightLine(FuncDefOp func, StringRef passName, StringRef funcName) {
  auto emitStraightLineError = [passName, funcName](Operation *op) {
    op->emitError() << passName << " expects a straight-line " << funcName
                    << " body; run `llzk-flatten` or another control-flow lowering pass first";
  };

  Region &body = func.getBody();
  if (!body.hasOneBlock()) {
    emitStraightLineError(func.getOperation());
    return failure();
  }

  Operation *unsupportedControlFlowOp = nullptr;
  body.walk([&](Operation *op) {
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      unsupportedControlFlowOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!unsupportedControlFlowOp) {
    return success();
  }

  emitStraightLineError(unsupportedControlFlowOp);
  return failure();
}

LogicalResult checkConstrainBodyIsStraightLine(FuncDefOp constrainFunc, StringRef passName) {
  return checkFuncBodyIsStraightLine(constrainFunc, passName, "constrain");
}

void replaceSubsequentUsesWith(Value oldVal, Value newVal, Operation *afterOp) {
  assert(afterOp && "afterOp must be a valid Operation*");

  for (auto &use : llvm::make_early_inc_range(oldVal.getUses())) {
    Operation *user = use.getOwner();

    // Skip uses that are:
    // - Before afterOp in the same block.
    // - Inside afterOp itself.
    if ((user->getBlock() == afterOp->getBlock()) &&
        (user == afterOp || user->isBeforeInBlock(afterOp))) {
      continue;
    }

    // Replace this use of oldVal with newVal.
    use.set(newVal);
  }
}

MemberDefOp addAuxMember(StructDefOp structDef, StringRef name, Type type) {
  assert(type && "auxiliary member type must be non-null");

  OpBuilder builder(structDef);
  builder.setInsertionPointToEnd(structDef.getBody());
  return builder.create<MemberDefOp>(structDef.getLoc(), builder.getStringAttr(name), type);
}

unsigned getFeltDegree(Value val, DenseMap<Value, unsigned> &memo) {
  if (auto it = memo.find(val); it != memo.end()) {
    return it->second;
  }

  if (isa<FeltConstantOp>(val.getDefiningOp())) {
    return memo[val] = 0;
  }
  if (isa<NonDetOp, MemberReadOp>(val.getDefiningOp()) || isa<BlockArgument>(val)) {
    return memo[val] = 1;
  }
  if (auto add = val.getDefiningOp<AddFeltOp>()) {
    return memo[val] =
               std::max(getFeltDegree(add.getLhs(), memo), getFeltDegree(add.getRhs(), memo));
  }
  if (auto sub = val.getDefiningOp<SubFeltOp>()) {
    return memo[val] =
               std::max(getFeltDegree(sub.getLhs(), memo), getFeltDegree(sub.getRhs(), memo));
  }
  if (auto mul = val.getDefiningOp<MulFeltOp>()) {
    return memo[val] = getFeltDegree(mul.getLhs(), memo) + getFeltDegree(mul.getRhs(), memo);
  }
  if (auto div = val.getDefiningOp<DivFeltOp>()) {
    return memo[val] = getFeltDegree(div.getLhs(), memo) + getFeltDegree(div.getRhs(), memo);
  }
  if (auto neg = val.getDefiningOp<NegFeltOp>()) {
    return memo[val] = getFeltDegree(neg.getOperand(), memo);
  }

  llvm::errs() << "Unhandled felt op in degree computation: " << val << '\n';
  llvm_unreachable("Unhandled op in getFeltDegree");
}

} // namespace llzk
