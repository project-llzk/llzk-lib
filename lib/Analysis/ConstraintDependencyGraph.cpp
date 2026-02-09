//===-- ConstraintDependencyGraph.cpp ---------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Analysis/DenseAnalysis.h"
#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Constrain/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Util/Hash.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Debug.h>

#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-cdg"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace constrain;
using namespace function;

/* SourceRefAnalysis */

void SourceRefAnalysis::visitCallControlFlowTransfer(
    mlir::CallOpInterface call, dataflow::CallControlFlowAction action,
    const SourceRefLattice &before, SourceRefLattice *after
) {
  LLVM_DEBUG(llvm::dbgs() << "SourceRefAnalysis::visitCallControlFlowTransfer: " << call << '\n');
  auto fnOpRes = resolveCallable<FuncDefOp>(tables, call);
  ensure(succeeded(fnOpRes), "could not resolve called function");

  LLVM_DEBUG({
    llvm::dbgs().indent(4) << "parent op is ";
    if (auto s = call->getParentOfType<StructDefOp>()) {
      llvm::dbgs() << s.getName();
    } else if (auto p = call->getParentOfType<FuncDefOp>()) {
      llvm::dbgs() << p.getName();
    } else {
      llvm::dbgs() << "<UNKNOWN PARENT TYPE>";
    }
    llvm::dbgs() << '\n';
  });

  /// `action == CallControlFlowAction::Enter` indicates that:
  ///   - `before` is the state before the call operation;
  ///   - `after` is the state at the beginning of the callee entry block;
  if (action == dataflow::CallControlFlowAction::EnterCallee) {
    // We skip updating the incoming lattice for function calls,
    // as SourceRefs are relative to the containing function/struct, so we don't need to pollute
    // the callee with the callers values.
    // This also avoids a non-convergence scenario, as calling a
    // function from other contexts can cause the lattice values to oscillate and constantly
    // change (thus looping infinitely).

    setToEntryState(after);
  }
  /// `action == CallControlFlowAction::Exit` indicates that:
  ///   - `before` is the state at the end of a callee exit block;
  ///   - `after` is the state after the call operation.
  else if (action == dataflow::CallControlFlowAction::ExitCallee) {
    // Get the argument values of the lattice by getting the state as it would
    // have been for the callsite.
    const SourceRefLattice *beforeCall = getLattice(getProgramPointBefore(call));
    ensure(beforeCall, "could not get prior lattice");

    // Translate argument values based on the operands given at the call site.
    std::unordered_map<SourceRef, SourceRefLatticeValue, SourceRef::Hash> translation;
    auto funcOpRes = resolveCallable<FuncDefOp>(tables, call);
    ensure(mlir::succeeded(funcOpRes), "could not lookup called function");
    auto funcOp = funcOpRes->get();

    auto callOp = llvm::dyn_cast<CallOp>(call.getOperation());
    ensure(callOp, "call is not a CallOp");

    for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
      SourceRef key(funcOp.getArgument(i));
      // Look up the lattice that defines the operand value first, but default
      // to the beforeCall if the operand is not defined by an operand.
      const SourceRefLattice *operandLattice = beforeCall;
      Value operand = callOp.getOperand(i);
      if (Operation *defOp = operand.getDefiningOp()) {
        operandLattice = getLattice(getProgramPointAfter(defOp));
      }

      translation[key] = operandLattice->getOrDefault(operand);
    }

    // The lattice at the return is the translated return values
    mlir::ChangeResult updated = mlir::ChangeResult::NoChange;
    for (unsigned i = 0; i < callOp.getNumResults(); i++) {
      auto retVal = before.getReturnValue(i);
      auto [translatedVal, _] = retVal.translate(translation);
      updated |= after->setValue(callOp->getResult(i), translatedVal);
    }
    propagateIfChanged(after, updated);
  }
  /// `action == CallControlFlowAction::External` indicates that:
  ///   - `before` is the state before the call operation.
  ///   - `after` is the state after the call operation, since there is no callee
  ///      body to enter into.
  else if (action == mlir::dataflow::CallControlFlowAction::ExternalCallee) {
    // For external calls, we propagate what information we already have from
    // before the call to after the call, since the external call won't invalidate
    // any of that information. It also, conservatively, makes no assumptions about
    // external calls and their computation, so CDG edges will not be computed over
    // input arguments to external functions.
    join(after, before);
  }
}

mlir::LogicalResult SourceRefAnalysis::visitOperation(
    mlir::Operation *op, const SourceRefLattice &before, SourceRefLattice *after
) {
  LLVM_DEBUG(llvm::dbgs() << "SourceRefAnalysis::visitOperation: " << *op << '\n');
  // Collect the references that are made by the operands to `op`.
  SourceRefLattice::ValueMap operandVals;
  for (OpOperand &operand : op->getOpOperands()) {
    const SourceRefLattice *prior = &before;
    // Lookup the lattice for the operand, if it is op defined.
    Value operandVal = operand.get();
    if (Operation *defOp = operandVal.getDefiningOp()) {
      prior = getLattice(getProgramPointAfter(defOp));
    }
    // Get the value (if there was a defining operation), or the default value.
    operandVals[operandVal] = prior->getOrDefault(operandVal);
  }

  // Add operand values, if not already added. Ensures that the default value
  // of a SourceRef (the source of the ref) is visible in the lattice.
  ChangeResult res = after->setValues(operandVals);

  // We will now join the the operand refs based on the type of operand.
  if (auto memberRefOp = llvm::dyn_cast<MemberRefOpInterface>(op)) {
    // The operand is indexed into by the MemberDefOp.
    auto memberOpRes = memberRefOp.getMemberDefOp(tables);
    ensure(mlir::succeeded(memberOpRes), "could not find member read");

    SourceRefLattice::ValueTy memberRefRes;
    if (memberRefOp.isRead()) {
      memberRefRes = memberRefOp.getVal();
    } else {
      memberRefRes = memberRefOp;
    }

    const auto &ops = operandVals.at(memberRefOp.getComponent());
    auto [memberVals, _] = ops.referenceMember(memberOpRes.value());

    res |= after->setValue(memberRefRes, memberVals);
  } else if (auto arrayAccessOp = llvm::dyn_cast<ArrayAccessOpInterface>(op)) {
    // Covers read/write/extract/insert array ops
    arraySubdivisionOpUpdate(arrayAccessOp, operandVals, before, after);
  } else if (auto createArray = llvm::dyn_cast<CreateArrayOp>(op)) {
    // Create an array using the operand values, if they exist.
    // Currently, the new array must either be fully initialized or uninitialized.
    SourceRefLatticeValue newArrayVal(createArray.getType().getShape());
    // If the array is statically initialized, iterate through all operands and initialize the array
    // value.
    const auto &elements = createArray.getElements();
    if (!elements.empty()) {
      for (unsigned i = 0; i < elements.size(); i++) {
        auto currentOp = elements[i];
        auto &opVals = operandVals[currentOp];
        (void)newArrayVal.getElemFlatIdx(i).setValue(opVals);
      }
    }

    auto createArrayRes = createArray.getResult();

    res |= after->setValue(createArrayRes, newArrayVal);
  } else if (auto structNewOp = llvm::dyn_cast<CreateStructOp>(op)) {
    auto newOpRes = structNewOp.getResult();
    auto newStructValue = before.getOrDefault(newOpRes);
    res |= after->setValue(newOpRes, newStructValue);
  } else {
    // Standard union of operands into the results value.
    // TODO: Could perform constant computation/propagation here for, e.g., arithmetic
    // over constants, but such analysis may be better suited for a dedicated pass.
    res |= fallbackOpUpdate(op, operandVals, before, after);
  }

  propagateIfChanged(after, res);
  LLVM_DEBUG(llvm::dbgs().indent(4) << "lattice is of size " << after->size() << '\n');
  return success();
}

// Perform a standard union of operands into the results value.
mlir::ChangeResult SourceRefAnalysis::fallbackOpUpdate(
    mlir::Operation *op, const SourceRefLattice::ValueMap &operandVals,
    const SourceRefLattice &before, SourceRefLattice *after
) {
  auto updated = mlir::ChangeResult::NoChange;
  for (auto res : op->getResults()) {
    auto cur = before.getOrDefault(res);

    for (auto &[_, opVal] : operandVals) {
      (void)cur.update(opVal);
    }
    updated |= after->setValue(res, cur);
  }
  return updated;
}

// Perform the update for either a readarr op or an extractarr op, which
// operate very similarly: index into the first operand using a variable number
// of provided indices.
void SourceRefAnalysis::arraySubdivisionOpUpdate(
    ArrayAccessOpInterface arrayAccessOp, const SourceRefLattice::ValueMap &operandVals,
    const SourceRefLattice & /*before*/, SourceRefLattice *after
) {
  // We index the first operand by all remaining indices.
  SourceRefLattice::ValueTy res;
  if (llvm::isa<ReadArrayOp, ExtractArrayOp>(arrayAccessOp)) {
    res = arrayAccessOp->getResult(0);
  } else {
    res = arrayAccessOp;
  }

  auto array = arrayAccessOp.getArrRef();
  auto it = operandVals.find(array);
  ensure(it != operandVals.end(), "improperly constructed operandVals map");
  auto currVals = it->second;

  std::vector<SourceRefIndex> indices;

  for (unsigned i = 0; i < arrayAccessOp.getIndices().size(); ++i) {
    auto idxOperand = arrayAccessOp.getIndices()[i];
    auto idxIt = operandVals.find(idxOperand);
    ensure(idxIt != operandVals.end(), "improperly constructed operandVals map");
    auto &idxVals = idxIt->second;

    // Note: we allow constant values regardless of if they are felt or index,
    // as if they were felt, there would need to be a cast to index, and if it
    // was missing, there would be a semantic check failure. So we accept either
    // so we don't have to track the cast ourselves.
    if (idxVals.isSingleValue() && idxVals.getSingleValue().isConstant()) {
      SourceRefIndex idx(idxVals.getSingleValue().getConstantValue());
      indices.push_back(idx);
    } else {
      // Otherwise, assume any range is valid.
      auto arrayType = llvm::dyn_cast<ArrayType>(array.getType());
      auto lower = mlir::APInt::getZero(64);
      mlir::APInt upper(64, arrayType.getDimSize(i));
      auto idxRange = SourceRefIndex(lower, upper);
      indices.push_back(idxRange);
    }
  }

  auto [newVals, _] = currVals.extract(indices);

  if (llvm::isa<ReadArrayOp, WriteArrayOp>(arrayAccessOp)) {
    ensure(newVals.isScalar(), "array read/write must produce a scalar value");
  }
  // an extract operation may yield a "scalar" value if not all dimensions of
  // the source array are instantiated; for example, if extracting an array from
  // an input arg, the current value is a "scalar" with an array type, and extracting
  // from that yields another single value with indices. For example: extracting [0][1]
  // from { arg1 } yields { arg1[0][1] }.

  propagateIfChanged(after, after->setValue(res, newVals));
}

/* ConstraintDependencyGraph */

mlir::FailureOr<ConstraintDependencyGraph> ConstraintDependencyGraph::compute(
    mlir::ModuleOp m, StructDefOp s, mlir::DataFlowSolver &solver, mlir::AnalysisManager &am,
    const CDGAnalysisContext &ctx
) {
  ConstraintDependencyGraph cdg(m, s, ctx);
  if (cdg.computeConstraints(solver, am).failed()) {
    return mlir::failure();
  }
  return cdg;
}

void ConstraintDependencyGraph::dump() const { print(llvm::errs()); }

/// Print all constraints. Any element that is unconstrained is omitted.
void ConstraintDependencyGraph::print(llvm::raw_ostream &os) const {
  // the EquivalenceClasses::iterator is sorted, but the EquivalenceClasses::member_iterator is
  // not guaranteed to be sorted. So, we will sort members before printing them.
  // We also want to add the constant values into the printing.
  std::set<std::set<SourceRef>> sortedSets;
  for (auto it = signalSets.begin(); it != signalSets.end(); it++) {
    if (!it->isLeader()) {
      continue;
    }

    std::set<SourceRef> sortedMembers;
    for (auto mit = signalSets.member_begin(it); mit != signalSets.member_end(); mit++) {
      sortedMembers.insert(*mit);
    }

    // We only want to print sets with a size > 1, because size == 1 means the
    // signal is not in a constraint.
    if (sortedMembers.size() > 1) {
      sortedSets.insert(sortedMembers);
    }
  }
  // Add the constants in separately.
  for (auto &[ref, constSet] : constantSets) {
    if (constSet.empty()) {
      continue;
    }
    std::set<SourceRef> sortedMembers(constSet.begin(), constSet.end());
    sortedMembers.insert(ref);
    sortedSets.insert(sortedMembers);
  }

  os << "ConstraintDependencyGraph { ";

  for (auto it = sortedSets.begin(); it != sortedSets.end();) {
    os << "\n    { ";
    for (auto mit = it->begin(); mit != it->end();) {
      os << *mit;
      mit++;
      if (mit != it->end()) {
        os << ", ";
      }
    }

    it++;
    if (it == sortedSets.end()) {
      os << " }\n";
    } else {
      os << " },";
    }
  }

  os << "}\n";
}

mlir::LogicalResult ConstraintDependencyGraph::computeConstraints(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am
) {
  // Fetch the constrain function. This is a required feature for all LLZK structs.
  FuncDefOp constrainFnOp = structDef.getConstrainFuncOp();
  ensure(
      constrainFnOp,
      "malformed struct " + mlir::Twine(structDef.getName()) + " must define a constrain function"
  );

  /**
   * Now, given the analysis, construct the CDG:
   * - Union all references based on solver results.
   * - Union all references based on nested dependencies.
   */

  // - Union all constraints from the analysis
  // This requires iterating over all of the emit operations
  constrainFnOp.walk([this, &solver](Operation *op) {
    ProgramPoint *pp = solver.getProgramPointAfter(op);
    const auto *refLattice = solver.lookupState<SourceRefLattice>(pp);
    // aggregate the ref2Val map across operations, as some may have nested
    // regions and blocks that aren't propagated to the function terminator
    if (refLattice) {
      for (auto &[ref, vals] : refLattice->getRef2Val()) {
        ref2Val[ref].insert(vals.begin(), vals.end());
      }
    }
    if (isa<EmitEqualityOp, EmitContainmentOp>(op)) {
      this->walkConstrainOp(solver, op);
    }
  });

  /**
   * Step two of the analysis is to traverse all of the constrain calls.
   * This is the nested analysis, basically.
   * Constrain functions don't return, so we don't need to compute "values" from
   * the call. We just need to see what constraints are generated here, and
   * add them to the transitive closures.
   */
  auto fnCallWalker = [this, &solver, &am](CallOp fnCall) mutable {
    auto res = resolveCallable<FuncDefOp>(tables, fnCall);
    ensure(mlir::succeeded(res), "could not resolve constrain call");

    auto fn = res->get();
    if (!fn.isStructConstrain()) {
      return;
    }
    // Nested
    auto calledStruct = fn.getOperation()->getParentOfType<StructDefOp>();
    SourceRefRemappings translations;

    ProgramPoint *pp = solver.getProgramPointAfter(fnCall.getOperation());
    auto *afterCallLattice = solver.lookupState<SourceRefLattice>(pp);
    ensure(afterCallLattice, "could not find lattice for call operation");

    // Map fn parameters to args in the call op
    for (unsigned i = 0; i < fn.getNumArguments(); i++) {
      SourceRef prefix(fn.getArgument(i));
      // Look up the lattice that defines the operand value first, but default
      // to the afterCallLattice if the operand is not defined by an operand.
      const SourceRefLattice *operandLattice = afterCallLattice;
      Value operand = fnCall.getOperand(i);
      if (Operation *defOp = operand.getDefiningOp()) {
        ProgramPoint *defPoint = solver.getProgramPointAfter(defOp);
        operandLattice = solver.lookupState<SourceRefLattice>(defPoint);
      }
      ensure(operandLattice, "could not find lattice for call operand");

      SourceRefLatticeValue val = operandLattice->getOrDefault(operand);
      translations.push_back({prefix, val});
    }
    auto &childAnalysis =
        am.getChildAnalysis<ConstraintDependencyGraphStructAnalysis>(calledStruct);
    if (!childAnalysis.constructed(ctx)) {
      ensure(
          mlir::succeeded(childAnalysis.runAnalysis(solver, am, {.runIntraprocedural = false})),
          "could not construct CDG for child struct"
      );
    }
    auto translatedCDG = childAnalysis.getResult(ctx).translate(translations);
    // Update the refMap with the translation
    const auto &translatedRef2Val = translatedCDG.getRef2Val();
    ref2Val.insert(translatedRef2Val.begin(), translatedRef2Val.end());

    // Now, union sets based on the translation
    // We should be able to just merge what is in the translatedCDG to the current CDG
    auto &tSets = translatedCDG.signalSets;
    for (auto lit = tSets.begin(); lit != tSets.end(); lit++) {
      if (!lit->isLeader()) {
        continue;
      }
      auto leader = lit->getData();
      for (auto mit = tSets.member_begin(lit); mit != tSets.member_end(); mit++) {
        signalSets.unionSets(leader, *mit);
      }
    }
    // And update the constant sets
    for (auto &[ref, constSet] : translatedCDG.constantSets) {
      constantSets[ref].insert(constSet.begin(), constSet.end());
    }
  };
  if (!ctx.runIntraproceduralAnalysis()) {
    constrainFnOp.walk(fnCallWalker);
  }

  return mlir::success();
}

void ConstraintDependencyGraph::walkConstrainOp(
    mlir::DataFlowSolver &solver, mlir::Operation *emitOp
) {
  std::vector<SourceRef> signalUsages, constUsages;

  ProgramPoint *pp = solver.getProgramPointAfter(emitOp);
  const SourceRefLattice *refLattice = solver.lookupState<SourceRefLattice>(pp);
  ensure(refLattice, "missing lattice for constrain op");

  for (auto operand : emitOp->getOperands()) {
    auto latticeVal = refLattice->getOrDefault(operand);
    for (auto &ref : latticeVal.foldToScalar()) {
      if (ref.isConstant()) {
        constUsages.push_back(ref);
      } else {
        signalUsages.push_back(ref);
      }
    }
  }

  // Compute a transitive closure over the signals.
  if (!signalUsages.empty()) {
    auto it = signalUsages.begin();
    auto leader = signalSets.getOrInsertLeaderValue(*it);
    for (it++; it != signalUsages.end(); it++) {
      signalSets.unionSets(leader, *it);
    }
  }
  // Also update constant references for each value.
  for (auto &sig : signalUsages) {
    constantSets[sig].insert(constUsages.begin(), constUsages.end());
  }
}

ConstraintDependencyGraph
ConstraintDependencyGraph::translate(SourceRefRemappings translation) const {
  ConstraintDependencyGraph res(mod, structDef, ctx);
  auto translate =
      [&translation](const SourceRef &elem) -> mlir::FailureOr<std::vector<SourceRef>> {
    std::vector<SourceRef> refs;
    for (auto &[prefix, vals] : translation) {
      if (!elem.isValidPrefix(prefix)) {
        continue;
      }

      if (vals.isArray()) {
        // Try to index into the array
        auto suffix = elem.getSuffix(prefix);
        ensure(
            mlir::succeeded(suffix), "failure is nonsensical, we already checked for valid prefix"
        );

        auto [resolvedVals, _] = vals.extract(suffix.value());
        auto folded = resolvedVals.foldToScalar();
        refs.insert(refs.end(), folded.begin(), folded.end());
      } else {
        for (auto &replacement : vals.getScalarValue()) {
          auto translated = elem.translate(prefix, replacement);
          if (mlir::succeeded(translated)) {
            refs.push_back(translated.value());
          }
        }
      }
    }
    if (refs.empty()) {
      return mlir::failure();
    }
    return refs;
  };

  for (auto leaderIt = signalSets.begin(); leaderIt != signalSets.end(); leaderIt++) {
    if (!leaderIt->isLeader()) {
      continue;
    }
    // translate everything in this set first
    std::vector<SourceRef> translatedSignals, translatedConsts;
    for (auto mit = signalSets.member_begin(leaderIt); mit != signalSets.member_end(); mit++) {
      auto member = translate(*mit);
      if (mlir::failed(member)) {
        continue;
      }
      for (auto &ref : *member) {
        if (ref.isConstant()) {
          translatedConsts.push_back(ref);
        } else {
          translatedSignals.push_back(ref);
        }
      }
      // Also add the constants from the original CDG
      if (auto it = constantSets.find(*mit); it != constantSets.end()) {
        auto &origConstSet = it->second;
        translatedConsts.insert(translatedConsts.end(), origConstSet.begin(), origConstSet.end());
      }
    }

    if (translatedSignals.empty()) {
      continue;
    }

    // Now we can insert the translated signals
    auto it = translatedSignals.begin();
    auto leader = *it;
    res.signalSets.insert(leader);
    for (it++; it != translatedSignals.end(); it++) {
      res.signalSets.insert(*it);
      res.signalSets.unionSets(leader, *it);
    }

    // And update the constant references
    for (auto &ref : translatedSignals) {
      res.constantSets[ref].insert(translatedConsts.begin(), translatedConsts.end());
    }
  }

  // Translate ref2Val as well
  for (auto &[ref, vals] : ref2Val) {
    auto translationRes = translate(ref);
    if (succeeded(translationRes)) {
      for (const auto &translatedRef : *translationRes) {
        res.ref2Val[translatedRef].insert(vals.begin(), vals.end());
      }
    }
  }

  return res;
}

SourceRefSet ConstraintDependencyGraph::getConstrainingValues(const SourceRef &ref) const {
  SourceRefSet res;
  auto currRef = mlir::FailureOr<SourceRef>(ref);
  while (mlir::succeeded(currRef)) {
    // Add signals
    for (auto it = signalSets.findLeader(*currRef); it != signalSets.member_end(); it++) {
      if (currRef.value() != *it) {
        res.insert(*it);
      }
    }
    // Add constants
    auto constIt = constantSets.find(*currRef);
    if (constIt != constantSets.end()) {
      res.insert(constIt->second.begin(), constIt->second.end());
    }
    // Go to parent
    currRef = currRef->getParentPrefix();
  }
  return res;
}

/* ConstraintDependencyGraphStructAnalysis */

mlir::LogicalResult ConstraintDependencyGraphStructAnalysis::runAnalysis(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &moduleAnalysisManager,
    const CDGAnalysisContext &ctx
) {
  auto result = ConstraintDependencyGraph::compute(
      getModule(), getStruct(), solver, moduleAnalysisManager, ctx
  );
  if (mlir::failed(result)) {
    return mlir::failure();
  }
  setResult(ctx, std::move(*result));
  return mlir::success();
}

} // namespace llzk
