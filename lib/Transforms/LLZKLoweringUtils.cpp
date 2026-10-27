//===-- LoweringUtils.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared utility function implementations for LLZK lowering passes.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/LLZK/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Ops.h"
#include "llzk/Transforms/LoweringUtils.h"

#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopeExit.h>
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

/// Whether a value can denote storage used by reconstructed aggregate reads.
bool isAggregateStorage(Value value) {
  return isa<array::ArrayType, pod::PodType, StructType>(value.getType());
}

/// Collects possible storage origins. Views, casts, and aggregate copies are all
/// conservatively associated with their source. Only entry arguments and known
/// allocations prove distinct origins; region forwarding and opaque producers
/// remain unknown instead of incorrectly proving disjointness.
bool collectStorageOrigins(Value value, DenseSet<Value> &origins, DenseSet<Value> &visited) {
  if (!visited.insert(value).second) {
    return true;
  }
  // Component values are identity-bearing handles, even when passed by value.
  // Different arguments or aggregate paths may refer to the same component.
  if (isa<StructType>(value.getType()) && !value.getDefiningOp<CreateStructOp>()) {
    return false;
  }
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    if (!isa<FuncDefOp>(argument.getOwner()->getParentOp()) ||
        argument.getOwner() != &argument.getOwner()->getParent()->front()) {
      return false;
    }
    origins.insert(value);
    return true;
  }
  Operation *def = value.getDefiningOp();
  if (isa<array::CreateArrayOp, pod::NewPodOp, CreateStructOp>(def)) {
    origins.insert(value);
    return true;
  }
  bool isAccess = isa<MemberReadOp, array::ArrayAccessOpInterface, pod::PodAccessOpInterface>(def);
  bool isAliasCast = isa<polymorphic::UnifiableCastOp, UnrealizedConversionCastOp>(def) &&
                     def->getNumOperands() == 1 && def->getNumResults() == 1;
  if (!isAccess && !isAliasCast) {
    return false;
  }
  bool hasSource = false;
  bool known = true;
  for (Value operand : def->getOperands()) {
    if (isAggregateStorage(operand)) {
      hasSource = true;
      known &= collectStorageOrigins(operand, origins, visited);
    }
  }
  return hasSource && known;
}

/// Proves disjointness only for distinct known origins, ignoring subpaths.
/// Treating value copies and different members as overlapping is conservative.
bool mayShareStorage(Value lhs, Value rhs) {
  DenseSet<Value> lhsOrigins, rhsOrigins, lhsVisited, rhsVisited;
  if (!collectStorageOrigins(lhs, lhsOrigins, lhsVisited) ||
      !collectStorageOrigins(rhs, rhsOrigins, rhsVisited)) {
    return true;
  }
  return llvm::any_of(lhsOrigins, [&](Value origin) { return rhsOrigins.contains(origin); });
}

/// Queries the body of a scope with an active-callee set for conservative recursion handling.
Operation *
findStorageMutation(Operation *scope, Value storage, DenseSet<Operation *> &activeCallees);

/// Checks each potentially overlapping call argument against the corresponding
/// callee argument. Opaque/external calls and recursive cycles cannot prove safety.
bool callMayMutateStorage(CallOp call, Value storage, DenseSet<Operation *> &activeCallees) {
  SmallVector<unsigned> overlappingArgs;
  for (auto [index, argument] : llvm::enumerate(call.getArgOperands())) {
    if (isAggregateStorage(argument) && mayShareStorage(argument, storage)) {
      overlappingArgs.push_back(index);
    }
  }
  if (overlappingArgs.empty()) {
    return false;
  }
  SymbolTableCollection tables;
  auto target = call.getCalleeTarget(tables);
  if (failed(target)) {
    return true;
  }
  FuncDefOp callee = target->get();
  if (callee.getBody().empty() || !activeCallees.insert(callee).second) {
    return true;
  }
  llvm::scope_exit cleanup([&]() { activeCallees.erase(callee); });
  for (unsigned index : overlappingArgs) {
    if (index >= callee.getNumArguments() ||
        findStorageMutation(callee, callee.getArgument(index), activeCallees)) {
      return true;
    }
  }
  return false;
}

/// Shared mutation model for both constrain reads and compute input mappings.
/// LLZK access interfaces describe their storage precisely. MLIR effects handle
/// other operations; unknown effects may mutate any aggregate operand. Calls
/// are analyzed using the same query on their callee arguments, without a whitelist.
bool operationMayMutateStorage(Operation *op, Value storage, DenseSet<Operation *> &activeCallees) {
  if (auto call = dyn_cast<CallOp>(op)) {
    return callMayMutateStorage(call, storage, activeCallees);
  }
  if (auto access = dyn_cast<array::ArrayAccessOpInterface>(op)) {
    return !access.isRead() && mayShareStorage(access.getArrRef(), storage);
  }
  if (auto access = dyn_cast<pod::PodAccessOpInterface>(op)) {
    return !access.isRead() && mayShareStorage(access.getPodRef(), storage);
  }
  if (auto write = dyn_cast<MemberWriteOp>(op)) {
    return mayShareStorage(write.getComponent(), storage);
  }
  // Constraint emission observes storage; terminators only transfer values.
  if (isa<ConstraintOpInterface>(op) || op->hasTrait<OpTrait::IsTerminator>()) {
    return false;
  }
  auto overlapsOperand = [&]() {
    return llvm::any_of(op->getOperands(), [&](Value operand) {
      return isAggregateStorage(operand) && mayShareStorage(operand, storage);
    });
  };
  auto effects = getEffectsRecursively(op);
  if (!effects) {
    return overlapsOperand();
  }
  return llvm::any_of(*effects, [&](const MemoryEffects::EffectInstance &effect) {
    if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect())) {
      return false;
    }
    if (Value target = effect.getValue()) {
      return !isAggregateStorage(target) || mayShareStorage(target, storage);
    }
    // An unscoped write effect cannot establish which storage it modifies.
    return true;
  });
}

/// Finds a possible mutation in any region/path of a scope. The same query is
/// used before constrain reads and across compute, where reads are replayed at returns.
Operation *
findStorageMutation(Operation *scope, Value storage, DenseSet<Operation *> &activeCallees) {
  Operation *mutation = nullptr;
  scope->walk([&](Operation *op) -> WalkResult {
    if (operationMayMutateStorage(op, storage, activeCallees)) {
      mutation = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return mutation;
}

/// Starts a storage-safety query without assumptions about previously visited callees.
Operation *findStorageMutation(Operation *scope, Value storage) {
  DenseSet<Operation *> activeCallees;
  return findStorageMutation(scope, storage, activeCallees);
}

Value mapBlockArgumentInCompute(BlockArgument barg, FuncDefOp computeFunc) {
  // Constrain entry arguments map onto compute inputs: constrain(%self, args...)
  // corresponds to compute(args...), plus the compute-side `%self`.
  if (barg.getArgNumber() == 0) {
    return computeFunc.getSelfValueFromCompute();
  }
  Value argument = computeFunc.getArgument(barg.getArgNumber() - 1);
  if (isAggregateStorage(argument) && findStorageMutation(computeFunc, argument)) {
    emitError(
        barg.getLoc()
    ) << "cannot rebuild aggregate argument in compute-side auxiliary expression: "
         "compute may mutate the input storage";
    return nullptr;
  }
  return argument;
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

  if (Operation *readOp = val.getDefiningOp();
      isa_and_nonnull<MemberReadOp, array::ReadArrayOp, array::ExtractArrayOp, pod::ReadPodOp>(
          readOp
      )) {
    // Replay is safe only if preceding operations cannot change the read's storage.
    // Use the same effect/provenance model as compute-side argument mappings.
    for (Operation &preceding : *readOp->getBlock()) {
      if (&preceding == readOp) {
        break;
      }
      for (Value operand : readOp->getOperands()) {
        if (isAggregateStorage(operand) && findStorageMutation(&preceding, operand)) {
          readOp->emitError(
              "cannot rebuild storage read in compute-side auxiliary expression: "
              "storage may have been mutated; aggregate mutations cannot be replayed"
          );
          return nullptr;
        }
      }
    }
    IRMapping mapper;
    for (Value operand : readOp->getOperands()) {
      Value rebuiltOperand = mapValueIntoCompute(operand, computeFunc, builder, memo);
      if (!rebuiltOperand) {
        return nullptr;
      }
      mapper.map(operand, rebuiltOperand);
    }

    Operation *rebuiltOp = builder.clone(*readOp, mapper);
    assert(rebuiltOp->getNumResults() == 1 && "storage reads have exactly one result");
    return memo[val] = rebuiltOp->getResult(0);
  }

  if (auto callOp = val.getDefiningOp<CallOp>()) {
    if (!callOp.getMapOperands().empty()) {
      callOp
          .emitError(
              "cannot rebuild affine-instantiated function.call in compute-side auxiliary "
              "expression"
          )
          .report();
      return nullptr;
    }

    SymbolTableCollection tables;
    FailureOr<SymbolLookupResult<FuncDefOp>> target = callOp.getCalleeTarget(tables);
    if (failed(target)) {
      return nullptr;
    }
    FuncDefOp targetFunc = target->get();
    bool invalidTarget =
        (targetFunc.hasAllowConstraintAttr() && !computeFunc.hasAllowConstraintAttr()) ||
        (targetFunc.hasAllowWitnessAttr() && !computeFunc.hasAllowWitnessAttr()) ||
        (targetFunc.hasAllowNonNativeFieldOpsAttr() &&
         !computeFunc.hasAllowNonNativeFieldOpsAttr());
    if (invalidTarget) {
      callOp
          .emitError(
              "cannot rebuild function.call in compute-side auxiliary expression: callee "
              "requires attributes not present on the compute function"
          )
          .report();
      return nullptr;
    }

    SmallVector<Value> rebuiltArgs;
    rebuiltArgs.reserve(callOp.getArgOperands().size());
    for (Value arg : callOp.getArgOperands()) {
      Value rebuiltArg = rebuildExprInCompute(arg, computeFunc, builder, memo);
      if (!rebuiltArg) {
        return nullptr;
      }
      rebuiltArgs.push_back(rebuiltArg);
    }

    ArrayRef<Attribute> templateParams;
    if (ArrayAttr params = callOp.getTemplateParamsAttr()) {
      templateParams = params.getValue();
    }

    CallOp rebuilt = CallOp::create(
        builder, callOp.getLoc(), callOp.getResultTypes(), callOp.getCalleeAttr(), rebuiltArgs,
        templateParams
    );
    for (auto [oldResult, newResult] : llvm::zip(callOp.getResults(), rebuilt.getResults())) {
      memo[oldResult] = newResult;
    }
    return memo[val];
  }

  if (val.getType().isIndex()) {
    // Preserve index producers used by member-read access operands so rebuilt reads
    // keep the original access semantics.
    Operation *defOp = val.getDefiningOp();
    assert(defOp && "index block arguments should already be mapped");

    IRMapping mapper;
    for (Value operand : defOp->getOperands()) {
      Value rebuiltOperand = mapValueIntoCompute(operand, computeFunc, builder, memo);
      if (!rebuiltOperand) {
        return nullptr;
      }
      mapper.map(operand, rebuiltOperand);
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
    if (!lhs || !rhs) {
      return nullptr;
    }
    return memo[val] = AddFeltOp::create(builder, add.getLoc(), add.getType(), lhs, rhs);
  }

  if (auto sub = val.getDefiningOp<SubFeltOp>()) {
    Value lhs = rebuildExprInCompute(sub.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(sub.getRhs(), computeFunc, builder, memo);
    if (!lhs || !rhs) {
      return nullptr;
    }
    return memo[val] = SubFeltOp::create(builder, sub.getLoc(), sub.getType(), lhs, rhs);
  }

  if (auto mul = val.getDefiningOp<MulFeltOp>()) {
    Value lhs = rebuildExprInCompute(mul.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(mul.getRhs(), computeFunc, builder, memo);
    if (!lhs || !rhs) {
      return nullptr;
    }
    return memo[val] = MulFeltOp::create(builder, mul.getLoc(), mul.getType(), lhs, rhs);
  }

  if (auto neg = val.getDefiningOp<NegFeltOp>()) {
    Value operand = rebuildExprInCompute(neg.getOperand(), computeFunc, builder, memo);
    if (!operand) {
      return nullptr;
    }
    return memo[val] = NegFeltOp::create(builder, neg.getLoc(), neg.getType(), operand);
  }

  if (auto div = val.getDefiningOp<DivFeltOp>()) {
    Value lhs = rebuildExprInCompute(div.getLhs(), computeFunc, builder, memo);
    Value rhs = rebuildExprInCompute(div.getRhs(), computeFunc, builder, memo);
    if (!lhs || !rhs) {
      return nullptr;
    }
    return memo[val] = DivFeltOp::create(builder, div.getLoc(), div.getType(), lhs, rhs);
  }

  if (auto c = val.getDefiningOp<FeltConstantOp>()) {
    return memo[val] = FeltConstantOp::create(builder, c.getLoc(), c.getValueAttr());
  }

  if (Operation *op = val.getDefiningOp()) {
    op->emitError("cannot rebuild unsupported operation in compute-side auxiliary expression")
        .report();
  }
  return nullptr;
}

LogicalResult checkForAuxMemberConflicts(StructDefOp structDef, StringRef prefix) {
  auto res = structDef.walk([&prefix](MemberDefOp memberDefOp) -> WalkResult {
    if (memberDefOp.getName().starts_with(prefix)) {
      return memberDefOp.emitOpError().append(
          "name conflicts with reserved prefix '", prefix, '\''
      );
    }
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
}

LogicalResult checkFuncBodyIsStraightLine(FuncDefOp func, StringRef passName) {
  auto emitStraightLineError = [&func, &passName](Operation *op) -> LogicalResult {
    StringRef funcName;
    if (func.isStructCompute()) {
      funcName = "compute";
    } else if (func.isStructConstrain()) {
      funcName = "constrain";
    } else {
      funcName = "function";
    }
    return op->emitError()
           << passName << " expects a straight-line " << funcName
           << " body; run `llzk-flatten` or another control-flow lowering pass first";
  };

  Region &body = func.getBody();
  if (!body.hasOneBlock()) {
    return emitStraightLineError(func.getOperation());
  }

  auto res = body.walk([&emitStraightLineError](Operation *op) -> WalkResult {
    if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0) {
      return emitStraightLineError(op);
    }
    return WalkResult::advance();
  });
  return failure(res.wasInterrupted());
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
  return MemberDefOp::create(builder, structDef.getLoc(), builder.getStringAttr(name), type);
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
