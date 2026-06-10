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
      result = analyzeCallResult(call, opRes.getResultNumber());
    } else if (auto ifOp = llvm::dyn_cast<scf::IfOp>(defOp)) {
      result = analyzeIfResult(ifOp, opRes.getResultNumber());
    } else if (auto forOp = llvm::dyn_cast<scf::ForOp>(defOp)) {
      result = analyzeForResult(forOp, opRes.getResultNumber());
    } else if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(defOp)) {
      result = analyzeWhileResult(whileOp, opRes.getResultNumber());
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

InfluenceInfo
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeBlockArgument(BlockArgument blockArg) {
  Block *owner = blockArg.getOwner();
  if (owner->isEntryBlock()) {
    return valueCache.lookup(blockArg);
  }

  Operation *parentOp = owner->getParentOp();
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(parentOp)) {
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber == 0) {
      return mergeInfluenceInfo(
          analyzeValue(forOp.getLowerBound()), analyzeValue(forOp.getUpperBound()),
          analyzeValue(forOp.getStep())
      );
    }
    unsigned iterIndex = argNumber - 1;
    return mergeInfluenceInfo(
        analyzeValue(forOp.getInitArgs()[iterIndex]), analyzeValue(forOp.getLowerBound()),

        analyzeValue(forOp.getUpperBound()),

        analyzeValue(forOp.getStep())
    );
  }
  if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(parentOp)) {
    Region *region = owner->getParent();
    if (region == &whileOp.getBefore()) {
      InfluenceInfo result = analyzeValue(whileOp.getInits()[blockArg.getArgNumber()]);
      Block &afterBlock = whileOp.getAfter().front();
      if (blockArg.getArgNumber() < afterBlock.getNumArguments()) {
        result = mergeInfluenceInfo(
            result, analyzeValue(afterBlock.getArgument(blockArg.getArgNumber()))
        );
      }
      return result;
    }
    if (region == &whileOp.getAfter()) {
      Block &beforeBlock = whileOp.getBefore().front();
      return analyzeValue(beforeBlock.getArgument(blockArg.getArgNumber()));
    }
  }
  return makeInfluenceInfo(Influence::None);
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeCallResult(
    CallOpInterface call, unsigned resultNumber
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
  return analyzer.analyzeCallableResult(resolvedCallable, argInfluences, resultNumber);
}

InfluenceInfo
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeIfResult(scf::IfOp ifOp, unsigned resultNumber) {
  InfluenceInfo result = analyzeValue(ifOp.getCondition());
  for (scf::YieldOp yieldOp : {ifOp.elseYield(), ifOp.thenYield()}) {
    if (yieldOp && yieldOp->getNumOperands() > resultNumber) {
      result = mergeInfluenceInfo(result, analyzeValue(yieldOp->getOperand(resultNumber)));
    }
  }
  return result;
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeForResult(
    scf::ForOp forOp, unsigned resultNumber
) {
  InfluenceInfo result = mergeInfluenceInfo(
      analyzeValue(forOp.getLowerBound()), analyzeValue(forOp.getUpperBound()),
      analyzeValue(forOp.getStep()), analyzeValue(forOp.getInitArgs()[resultNumber])
  );
  ValueRange yieldedValues = forOp.getYieldedValues();
  if (yieldedValues.size() > resultNumber) {
    result = mergeInfluenceInfo(result, analyzeValue(yieldedValues[resultNumber]));
  }
  return result;
}

InfluenceInfo ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeWhileResult(
    scf::WhileOp whileOp, unsigned resultNumber
) {
  InfluenceInfo result = analyzeValue(whileOp.getInits()[resultNumber]);
  scf::ConditionOp condOp = whileOp.getConditionOp();
  result = mergeInfluenceInfo(result, analyzeValue(condOp.getCondition()));
  if (condOp.getArgs().size() > resultNumber) {
    result = mergeInfluenceInfo(result, analyzeValue(condOp.getArgs()[resultNumber]));
  }
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  if (yieldOp.getNumOperands() > resultNumber) {
    result = mergeInfluenceInfo(result, analyzeValue(yieldOp.getOperand(resultNumber)));
  }
  return result;
}

//===------------------------------------------------------------------===//
// ForbiddenInfluenceAnalyzer
//===------------------------------------------------------------------===//

InfluenceInfo ForbiddenInfluenceAnalyzer::analyzeContractValue(ContractOp contract, Value value) {
  llvm::SmallVector<InfluenceInfo> argInfluenceInfos;
  for (BlockArgument arg : contract.getArguments()) {
    argInfluenceInfos.push_back(classifyContractArgument(contract, arg));
  }
  ForbiddenInfluenceAnalyzer::AnalysisFrame frame(*this, contract, argInfluenceInfos);
  return frame.analyzeValue(value);
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
    llvm::SmallVector<ReturnOp> returnOps;
    region->walk([&](ReturnOp op) { returnOps.push_back(op); });
    for (ReturnOp retOp : returnOps) {
      if (retOp.getNumOperands() > resultNumber) {
        summary = mergeInfluenceInfo(summary, frame.analyzeValue(retOp.getOperand(resultNumber)));
      }
    }
  }

  activeSummaries.erase(key);
  callableSummaryCache[key] = summary;
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
