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

//===------------------------------------------------------------------===//
// ForbiddenInfluenceAnalyzer::AnalysisFrame
//===------------------------------------------------------------------===//

ForbiddenInfluenceAnalyzer::AnalysisFrame::AnalysisFrame(
    ForbiddenInfluenceAnalyzer &parentAnalyzer, CallableOpInterface callableOp,
    llvm::ArrayRef<Influence> argInfluences
)
    : analyzer(parentAnalyzer) {
  Region *region = callableOp.getCallableRegion();
  assert(region && !region->empty() && "callable must have a body");
  Block &entry = region->front();
  for (auto [arg, influence] : llvm::zip(entry.getArguments(), argInfluences)) {
    valueCache[arg] = influence;
  }
}

Influence ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeValue(Value value) {
  if (auto it = valueCache.find(value); it != valueCache.end()) {
    return it->second;
  }
  if (!activeValues.insert(value).second) {
    return Influence::None;
  }

  Influence result = Influence::None;
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(value)) {
    result = analyzeBlockArgument(blockArg);
  } else if (Operation *defOp = value.getDefiningOp()) {
    auto opRes = llvm::dyn_cast<OpResult>(value);
    assert(opRes && "value has defining op, so it must be an op result");
    if (llvm::isa<MemberReadOp>(defOp)) {
      result = Influence::StructMember;
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
        result |= analyzeValue(operand);
      }
    }
  }

  activeValues.erase(value);
  valueCache[value] = result;
  return result;
}

Influence ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeBlockArgument(BlockArgument blockArg) {
  Block *owner = blockArg.getOwner();
  if (owner->isEntryBlock()) {
    return valueCache.lookup(blockArg);
  }

  Operation *parentOp = owner->getParentOp();
  if (auto forOp = llvm::dyn_cast<scf::ForOp>(parentOp)) {
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber == 0) {
      return analyzeValue(forOp.getLowerBound()) | analyzeValue(forOp.getUpperBound()) |
             analyzeValue(forOp.getStep());
    }
    unsigned iterIndex = argNumber - 1;
    return analyzeValue(forOp.getInitArgs()[iterIndex]) | analyzeValue(forOp.getLowerBound()) |
           analyzeValue(forOp.getUpperBound()) | analyzeValue(forOp.getStep());
  }
  if (auto whileOp = llvm::dyn_cast<scf::WhileOp>(parentOp)) {
    Region *region = owner->getParent();
    if (region == &whileOp.getBefore()) {
      Influence result = analyzeValue(whileOp.getInits()[blockArg.getArgNumber()]);
      Block &afterBlock = whileOp.getAfter().front();
      if (blockArg.getArgNumber() < afterBlock.getNumArguments()) {
        result |= analyzeValue(afterBlock.getArgument(blockArg.getArgNumber()));
      }
      return result;
    }
    if (region == &whileOp.getAfter()) {
      Block &beforeBlock = whileOp.getBefore().front();
      return analyzeValue(beforeBlock.getArgument(blockArg.getArgNumber()));
    }
  }
  return Influence::None;
}

Influence ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeCallResult(
    CallOpInterface call, unsigned resultNumber
) {
  auto resolvedCallable = llvm::dyn_cast_if_present<CallableOpInterface>(call.resolveCallable());
  if (!resolvedCallable || !resolvedCallable.getCallableRegion()) {
    return Influence::FunctionReturn;
  }

  llvm::SmallVector<Influence> argInfluences;
  argInfluences.reserve(call.getArgOperands().size());
  for (Value operand : call.getArgOperands()) {
    argInfluences.push_back(analyzeValue(operand));
  }
  return analyzer.analyzeCallableResult(resolvedCallable, argInfluences, resultNumber);
}

Influence
ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeIfResult(scf::IfOp ifOp, unsigned resultNumber) {
  Influence result = analyzeValue(ifOp.getCondition());
  for (scf::YieldOp yieldOp : {ifOp.elseYield(), ifOp.thenYield()}) {
    if (yieldOp && yieldOp->getNumOperands() > resultNumber) {
      result |= analyzeValue(yieldOp->getOperand(resultNumber));
    }
  }
  return result;
}

Influence ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeForResult(
    scf::ForOp forOp, unsigned resultNumber
) {
  Influence result = analyzeValue(forOp.getLowerBound()) | analyzeValue(forOp.getUpperBound()) |
                     analyzeValue(forOp.getStep()) |
                     analyzeValue(forOp.getInitArgs()[resultNumber]);
  ValueRange yieldedValues = forOp.getYieldedValues();
  if (yieldedValues.size() > resultNumber) {
    result |= analyzeValue(yieldedValues[resultNumber]);
  }
  return result;
}

Influence ForbiddenInfluenceAnalyzer::AnalysisFrame::analyzeWhileResult(
    scf::WhileOp whileOp, unsigned resultNumber
) {
  Influence result = analyzeValue(whileOp.getInits()[resultNumber]);
  scf::ConditionOp condOp = whileOp.getConditionOp();
  result |= analyzeValue(condOp.getCondition());
  if (condOp.getArgs().size() > resultNumber) {
    result |= analyzeValue(condOp.getArgs()[resultNumber]);
  }
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  if (yieldOp.getNumOperands() > resultNumber) {
    result |= analyzeValue(yieldOp.getOperand(resultNumber));
  }
  return result;
}

//===------------------------------------------------------------------===//
// ForbiddenInfluenceAnalyzer
//===------------------------------------------------------------------===//

Influence ForbiddenInfluenceAnalyzer::analyzeContractValue(ContractOp contract, Value value) {
  llvm::SmallVector<Influence> argInfluences;
  for (BlockArgument arg : contract.getArguments()) {
    argInfluences.push_back(classifyContractArgument(contract, arg));
  }
  ForbiddenInfluenceAnalyzer::AnalysisFrame frame(*this, contract, argInfluences);
  return frame.analyzeValue(value);
}

Influence ForbiddenInfluenceAnalyzer::analyzeCallableResult(
    CallableOpInterface callableOp, llvm::ArrayRef<Influence> argInfluences, unsigned resultNumber
) {
  CallableSummaryKey key {
      .callable = callableOp,
      .argInfluences = llvm::SmallVector<uint8_t>(argInfluences.size()),
      .resultNumber = resultNumber,
  };
  for (auto [idx, influence] : llvm::enumerate(argInfluences)) {
    key.argInfluences[idx] = static_cast<uint8_t>(influence);
  }

  if (auto it = callableSummaryCache.find(key); it != callableSummaryCache.end()) {
    return it->second;
  }
  if (!activeSummaries.insert(key).second) {
    return Influence::FunctionReturn;
  }

  Influence summary = Influence::FunctionReturn;
  Region *region = callableOp.getCallableRegion();
  if (region && !region->empty()) {
    AnalysisFrame frame(*this, callableOp, argInfluences);
    summary = Influence::None;
    llvm::SmallVector<ReturnOp> returnOps;
    region->walk([&](ReturnOp op) { returnOps.push_back(op); });
    for (ReturnOp retOp : returnOps) {
      if (retOp.getNumOperands() > resultNumber) {
        summary |= frame.analyzeValue(retOp.getOperand(resultNumber));
      }
    }
  }

  activeSummaries.erase(key);
  callableSummaryCache[key] = summary;
  return summary;
}

Influence
ForbiddenInfluenceAnalyzer::classifyContractArgument(ContractOp contract, BlockArgument arg) {
  SymbolTableCollection tables;
  auto funcTarget = contract.getFuncTarget(tables);
  if (failed(funcTarget)) {
    return Influence::None;
  }

  unsigned numFuncInputs = funcTarget->get().getFunctionType().getNumInputs();
  if (arg.getArgNumber() >= numFuncInputs) {
    return Influence::FunctionReturn;
  }
  return Influence::None;
}
