//===-- ForbiddenPreconditionInfluence.cpp ----------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/Util/ForbiddenPreconditionInfluence.h"

#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

using namespace mlir;
using namespace llzk::component;
using namespace llzk::function;
using namespace llzk::verif;
using namespace llzk::verif::detail;

static InfluenceInfo
makeInfluenceInfo(Influence influence, std::optional<Location> loc = std::nullopt) {
  InfluenceInfo info;
  info.influence = influence;
  if (loc.has_value()) {
    info.structMemberLocs.insert(loc.value());
  }
  return info;
}

//===------------------------------------------------------------------===//
// ForbiddenInfluenceAnalyzer::AnalysisFrame
//===------------------------------------------------------------------===//

ForbiddenInfluenceAnalyzer::AnalysisFrame::AnalysisFrame(
    ForbiddenInfluenceAnalyzer &parentAnalyzer, CallableOpInterface callableOp,
    llvm::ArrayRef<InfluenceInfo> argInfluenceInfos
)
    : analyzer(parentAnalyzer) {
  Region *region = callableOp.getCallableRegion();
  assert(region && !region->empty() && "callable must have a body");
  Block &entry = region->front();
  for (auto [arg, influenceInfo] : llvm::zip(entry.getArguments(), argInfluenceInfos)) {
    valueCache[arg] = influenceInfo;
  }
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeValue(Value value) {
  if (auto it = valueCache.find(value); it != valueCache.end()) {
    return it->second;
  }
  if (!activeValues.insert(value).second) {
    return makeInfluenceInfo(Influence::None);
  }

  InfluenceInfo result = makeInfluenceInfo(Influence::None);
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
    result = analyzeBlockArgument(blockArg);
  } else if (Operation *defOp = value.getDefiningOp()) {
    auto opRes = llvm::dyn_cast<OpResult>(value);
    assert(opRes && "value has defining op, so it must be an op result");
    if (llvm::isa<MemberReadOp>(defOp)) {
      result = makeInfluenceInfo(Influence::StructMember, defOp->getLoc());
    } else if (auto call = llvm::dyn_cast<CallOpInterface>(defOp)) {
      result = analyzeCallResult(call, opRes);
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(defOp)) {
      result = analyzeIfResult(ifOp, opRes);
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(defOp)) {
      result = analyzeForResult(forOp, opRes);
    } else if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(defOp)) {
      result = analyzeWhileResult(whileOp, opRes);
    } else {
      for (Value operand : defOp->getOperands()) {
        result = mergeInfluenceInfo(result, analyzeValue(operand));
      }
    }
  }

  activeValues.erase(value);
  valueCache[value] = result;
  return result;
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzePreconditionOp(
    PreconditionOpInterface preCondOp
) {
  return mergeInfluenceInfo(
      analyzeValue(preCondOp.getCondition()), analyzeControlAncestors(preCondOp.getOperation())
  );
}

IncludedContractSummary
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeIncludeOp(IncludeOp includeOp) {
  SymbolTableCollection tables;
  auto calleeTarget = includeOp.getCalleeTarget(tables);
  if (failed(calleeTarget)) {
    IncludedContractSummary summary;
    summary.failures.push_back(
        {.precondition = {}, .influenceInfo = makeInfluenceInfo(Influence::FunctionReturn)}
    );
    return summary;
  }

  ContractOp calleeContract = calleeTarget->get();

  llvm::SmallVector<InfluenceInfo> argInfluences;
  argInfluences.reserve(includeOp.getArgOperands().size());
  for (Value operand : includeOp.getArgOperands()) {
    argInfluences.push_back(analyzeValue(operand));
  }
  return analyzer.analyzeIncludedContract(calleeContract, argInfluences);
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeControlAncestors(Operation *op) {
  if (auto it = controlAncestorCache.find(op); it != controlAncestorCache.end()) {
    return it->second;
  }

  InfluenceInfo result = makeInfluenceInfo(Influence::None);
  Operation *current = op;
  while (Operation *parentOp = current->getParentOp()) {
    if (isa<ContractOp>(parentOp)) {
      break;
    }
    result = mergeInfluenceInfo(result, analyzeAncestorControl(parentOp, current));
    current = parentOp;
  }

  controlAncestorCache[op] = result;
  return result;
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeAncestorControl(
    Operation *ancestor, Operation *nestedOp
) {
  if (auto ifOp = dyn_cast<scf::IfOp>(ancestor)) {
    return analyzeValue(ifOp.getCondition());
  }
  if (auto forOp = dyn_cast<scf::ForOp>(ancestor)) {
    return mergeInfluenceInfo(
        analyzeValue(forOp.getLowerBound()), analyzeValue(forOp.getUpperBound()),
        analyzeValue(forOp.getStep())
    );
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(ancestor)) {
    InfluenceInfo result = analyzeValue(whileOp.getConditionOp().getCondition());
    Region *nestedRegion = nestedOp->getParentRegion();
    if (nestedRegion == &whileOp.getAfter()) {
      unsigned beforeArgCount = whileOp.getBefore().front().getNumArguments();
      for (unsigned i = 0; i < beforeArgCount; ++i) {
        result =
            mergeInfluenceInfo(result, analyzeValue(whileOp.getBefore().front().getArgument(i)));
      }
    }
    return result;
  }
  return makeInfluenceInfo(Influence::None);
}

InfluenceInfo
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeBlockArgument(BlockArgument blockArg) {
  Block *owner = blockArg.getOwner();

  Operation *parentOp = owner->getParentOp();
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(parentOp)) {
    unsigned argNumber = blockArg.getArgNumber();
    InfluenceInfo tripCountInfo = mergeInfluenceInfo(
        analyzeValue(forOp.getLowerBound()), analyzeValue(forOp.getUpperBound()),
        analyzeValue(forOp.getStep())
    );
    // arg0 is the induction var, influenced by the lower bound, upper bound, and
    // the step operands to the loop.
    // The other for loop region arguments are additionally influenced by the
    // init args and yield args. They are also conservatively influenced by the loop trip count
    // values (bounds and step).
    if (argNumber != 0) {
      unsigned iterIndex = argNumber - 1;
      return mergeInfluenceInfo(
          tripCountInfo, analyzeValue(forOp.getInitArgs()[iterIndex]),
          analyzeValue(forOp.getYieldedValues()[iterIndex])
      );
    }
    return tripCountInfo;
  }
  if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(parentOp)) {
    Region *region = owner->getParent();
    if (region == &whileOp.getBefore()) {
      // Depends both on the init args and yield values
      return mergeInfluenceInfo(
          analyzeValue(whileOp.getInits()[blockArg.getArgNumber()]),
          analyzeValue(whileOp.getYieldOp().getResults()[blockArg.getArgNumber()])
      );
    }
    if (region == &whileOp.getAfter()) {
      // The after block is given arguments from the condition op. These arguments
      // also don't have to align with the before region args, which is confusing at
      // first glance. https://mlir.llvm.org/docs/Dialects/SCFDialect/#scfwhile-scfwhileop
      scf::ConditionOp condOp = whileOp.getConditionOp();
      auto condArgs = condOp.getArgs();
      assert(
          condArgs.size() == region->getNumArguments() &&
          "condition args must equal after region args"
      );
      return analyzeValue(condArgs[blockArg.getArgNumber()]);
    }
  }
  return makeInfluenceInfo(Influence::None);
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeCallResult(
    CallOpInterface call, OpResult callRes
) {
  auto resolvedCallable = llvm::dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  if (!resolvedCallable || !resolvedCallable.getCallableRegion()) {
    return makeInfluenceInfo(Influence::FunctionReturn);
  }

  llvm::SmallVector<InfluenceInfo> argInfluences;
  argInfluences.reserve(call.getArgOperands().size());
  for (Value operand : call.getArgOperands()) {
    argInfluences.push_back(analyzeValue(operand));
  }
  return analyzer.analyzeCallableResult(resolvedCallable, argInfluences, callRes.getResultNumber());
}

InfluenceInfo
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeIfResult(scf::IfOp ifOp, OpResult ifRes) {
  unsigned resultNumber = ifRes.getResultNumber();
  InfluenceInfo result = analyzeValue(ifOp.getCondition());
  for (scf::YieldOp yieldOp : {ifOp.elseYield(), ifOp.thenYield()}) {
    if (yieldOp && yieldOp->getNumOperands() > resultNumber) {
      result = mergeInfluenceInfo(result, analyzeValue(yieldOp->getOperand(resultNumber)));
    }
  }
  return result;
}

InfluenceInfo
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeForResult(scf::ForOp forOp, OpResult forRes) {
  unsigned resultNumber = forRes.getResultNumber();
  return mergeInfluenceInfo(
      analyzeValue(forOp.getLowerBound()), analyzeValue(forOp.getUpperBound()),
      analyzeValue(forOp.getStep()), analyzeValue(forOp.getInitArgs()[resultNumber]),
      analyzeValue(forOp.getYieldedValues()[resultNumber])
  );
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeWhileResult(
    scf::WhileOp whileOp, OpResult whileRes
) {
  unsigned resultNumber = whileRes.getResultNumber();
  // The number of results must match, by definition:
  // - size of init args
  // - size of condition op args (excluding the condition value)
  // - size of yield operands
  auto inits = whileOp.getInits();
  scf::ConditionOp condOp = whileOp.getConditionOp();
  auto condArgs = condOp.getArgs();
  auto yieldOps = whileOp.getYieldOp().getOperands();
  assert(
      inits.size() == condArgs.size() && condArgs.size() == yieldOps.size() &&
      "invalid while op dimensions"
  );
  assert(inits.size() > resultNumber && "invalid result number");

  return mergeInfluenceInfo(
      analyzeValue(inits[resultNumber]), analyzeValue(condArgs[resultNumber]),
      analyzeValue(yieldOps[resultNumber]),
      // The results are also control-flow influenced by the condition value itself.
      analyzeValue(condOp.getCondition())
  );
}

//===------------------------------------------------------------------===//
// ForbiddenInfluenceAnalyzer
//===------------------------------------------------------------------===//

InfluenceInfo ForbiddenInfluenceAnalyzer::analyzeContractValue(ContractOp contract, Value value) {
  if (auto it = cachedFrames.find(contract); it != cachedFrames.end()) {
    return it->second.analyzeValue(value);
  }
  llvm::SmallVector<InfluenceInfo> argInfluenceInfos;
  for (BlockArgument arg : contract.getArguments()) {
    argInfluenceInfos.push_back(classifyContractArgument(contract, arg));
  }
  auto [it, inserted] = cachedFrames.try_emplace(contract, *this, contract, argInfluenceInfos);
  assert(inserted && "lookup failure");
  return it->second.analyzeValue(value);
}

InfluenceInfo ForbiddenInfluenceAnalyzer::analyzePreconditionOp(
    ContractOp contract, PreconditionOpInterface preCondOp
) {
  if (auto it = cachedFrames.find(contract); it != cachedFrames.end()) {
    return it->second.analyzePreconditionOp(preCondOp);
  }

  llvm::SmallVector<InfluenceInfo> argInfluenceInfos;
  for (BlockArgument arg : contract.getArguments()) {
    argInfluenceInfos.push_back(classifyContractArgument(contract, arg));
  }
  auto [it, inserted] = cachedFrames.try_emplace(contract, *this, contract, argInfluenceInfos);
  assert(inserted && "lookup failure");
  return it->second.analyzePreconditionOp(preCondOp);
}

InfluenceInfo ForbiddenInfluenceAnalyzer::analyzeCallableResult(
    CallableOpInterface callableOp, llvm::ArrayRef<InfluenceInfo> argInfluences,
    unsigned resultNumber
) {
  CallableSummaryKey key {
      .callable = callableOp,
      .argInfluences = llvm::SmallVector<InfluenceInfo>(argInfluences.begin(), argInfluences.end()),
      .resultNumber = resultNumber,
  };

  if (auto it = callableSummaryCache.find(key); it != callableSummaryCache.end()) {
    return it->second;
  }
  if (!activeSummaries.insert(key).second) {
    return makeInfluenceInfo(Influence::FunctionReturn);
  }

  InfluenceInfo summary = makeInfluenceInfo(Influence::FunctionReturn);
  Region *region = callableOp.getCallableRegion();
  if (region && !region->empty()) {
    AnalysisFrame frame(*this, callableOp, argInfluences);
    summary = makeInfluenceInfo(Influence::None);
    region->walk([&](ReturnOp retOp) {
      if (retOp.getNumOperands() > resultNumber) {
        summary = mergeInfluenceInfo(summary, frame.analyzeValue(retOp.getOperand(resultNumber)));
      }
    });
  }

  activeSummaries.erase(key);
  callableSummaryCache[key] = summary;
  return summary;
}

IncludedContractSummary ForbiddenInfluenceAnalyzer::analyzeIncludedContract(
    ContractOp calleeContract, llvm::ArrayRef<InfluenceInfo> argInfluences
) {
  IncludedContractSummaryKey key {
      .contract = calleeContract,
      .argInfluences = llvm::SmallVector<InfluenceInfo>(argInfluences.begin(), argInfluences.end()),
  };

  if (auto it = includedContractSummaryCache.find(key); it != includedContractSummaryCache.end()) {
    return it->second;
  }
  if (!activeIncludedSummaries.insert(key).second) {
    IncludedContractSummary summary;
    summary.failures.push_back(
        {.precondition = {}, .influenceInfo = makeInfluenceInfo(Influence::FunctionReturn)}
    );
    return summary;
  }

  AnalysisFrame frame(*this, calleeContract, argInfluences);
  IncludedContractSummary summary;

  SmallVector<PreconditionOpInterface> preconditionOps;
  calleeContract.walk([&](PreconditionOpInterface op) { preconditionOps.push_back(op); });
  for (PreconditionOpInterface preCondOp : preconditionOps) {
    InfluenceInfo influenceInfo = frame.analyzePreconditionOp(preCondOp);
    if (any(influenceInfo.influence)) {
      summary.failures.push_back({.precondition = preCondOp, .influenceInfo = influenceInfo});
    }
  }

  SmallVector<IncludeOp> includeOps;
  calleeContract.walk([&](IncludeOp includeOp) { includeOps.push_back(includeOp); });
  for (IncludeOp includeOp : includeOps) {
    IncludedContractSummary nestedSummary = frame.analyzeIncludeOp(includeOp);
    summary.failures.append(nestedSummary.failures.begin(), nestedSummary.failures.end());
  }

  activeIncludedSummaries.erase(key);
  includedContractSummaryCache[key] = summary;
  return summary;
}

InfluenceInfo
ForbiddenInfluenceAnalyzer::classifyContractArgument(ContractOp contract, BlockArgument arg) {
  SymbolTableCollection tables;
  auto funcTarget = contract.getFuncTarget(tables);
  if (failed(funcTarget)) {
    return makeInfluenceInfo(Influence::None);
  }

  unsigned numFuncInputs = funcTarget->get().getFunctionType().getNumInputs();
  if (arg.getArgNumber() >= numFuncInputs) {
    return makeInfluenceInfo(Influence::FunctionReturn);
  }
  return makeInfluenceInfo(Influence::None);
}
