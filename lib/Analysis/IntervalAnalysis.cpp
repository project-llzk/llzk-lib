//===-- IntervalAnalysis.cpp - Interval analysis implementation -*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/IntervalAnalysis.h"

#include "llzk/Analysis/AnalysisUtil.h"
#include "llzk/Analysis/Matchers.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Array/Util/ArrayTypeHelper.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/StreamHelper.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/TypeSwitch.h>

#include <functional>
#include <limits>
#include <map>

using namespace mlir;

namespace llzk {

using namespace array;
using namespace boolean;
using namespace cast;
using namespace component;
using namespace constrain;
using namespace felt;
using namespace function;

namespace {

/// Allocate a namespace that is never reused while the SMT solver remains alive.
/// Weak ownership keeps independent solvers independent without extending their lifetimes.
/// In particular, destroying an analysis must not reset a still-live solver's counter.
std::string allocateSymbolNamespace(const llvm::SMTSolverRef &solver) {
  using WeakSolver = std::weak_ptr<llvm::SMTSolver>;
  static std::map<WeakSolver, uint64_t, std::owner_less<WeakSolver>> nextIDs;
  static std::mutex mutex;
  std::lock_guard lock(mutex);
  std::erase_if(nextIDs, [](const auto &entry) { return entry.first.expired(); });
  auto &nextID = nextIDs[solver];
  ensure(nextID != std::numeric_limits<uint64_t>::max(), "SMT symbol namespace exhausted");
  return "llzk.interval." + std::to_string(nextID++) + ".";
}

std::optional<UnreducedInterval> mergeUnreducedIntervals(
    const std::optional<UnreducedInterval> &lhs, const std::optional<UnreducedInterval> &rhs
) {
  if (!lhs.has_value() || !rhs.has_value()) {
    return std::nullopt;
  }
  return lhs->doUnion(*rhs);
}

/// Match a storage prefix using overlapping array selections in either direction.
bool hasOverlappingPrefix(const SourceRef &ref, const SourceRef &prefix) {
  if (!ref.isRooted() || !prefix.isRooted() || *ref.getRoot() != *prefix.getRoot() ||
      ref.getPath().size() < prefix.getPath().size()) {
    return false;
  }
  return llvm::all_of(
      llvm::zip(ref.getPath().take_front(prefix.getPath().size()), prefix.getPath()),
      [](const auto &indices) { return std::get<0>(indices).overlaps(std::get<1>(indices)); }
  );
}

/// Ranges describe possible locations rather than one concrete storage identity.
bool hasRangedSelection(const SourceRef &ref) {
  return llvm::any_of(ref.getPath(), [](const SourceRefIndex &index) {
    return index.isIndexRange();
  });
}

/// Prove same-block reads equivalent using their SSA indices, without crossing possible writes.
bool haveSameReadValue(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  Operation *left = lhs.getDefiningOp(), *right = rhs.getDefiningOp();
  if (!llvm::isa_and_present<ReadArrayOp, MemberReadOp>(left) ||
      !llvm::isa_and_present<ReadArrayOp, MemberReadOp>(right) ||
      left->getBlock() != right->getBlock()) {
    return false;
  }
  auto sameOperand = [](Value a, Value b) {
    return success(
        a == b || (llvm::isa<ArrayType, StructType>(a.getType()) && haveSameReadValue(a, b))
    );
  };
  if (!OperationEquivalence::isEquivalentTo(
          left, right, sameOperand, nullptr, OperationEquivalence::IgnoreLocations
      )) {
    return false;
  }
  if (right->isBeforeInBlock(left)) {
    std::swap(left, right);
  }
  for (Operation *between = left->getNextNode(); between != right;
       between = between->getNextNode()) {
    auto arrayAccess = llvm::dyn_cast<ArrayAccessOpInterface>(between);
    if (arrayAccess && arrayAccess.isRead()) {
      continue;
    }
    if (!isMemoryEffectFree(between) && !hasSingleEffect<MemoryEffects::Read>(between)) {
      return false;
    }
  }
  return true;
}

template <typename Fn>
std::optional<UnreducedInterval>
combineUnreducedIntervals(const ExpressionValue &lhs, const ExpressionValue &rhs, Fn &&fn) {
  if (!lhs.hasUnreducedInterval() || !rhs.hasUnreducedInterval()) {
    return std::nullopt;
  }
  return fn(lhs.getUnreducedInterval(), rhs.getUnreducedInterval());
}

ExpressionValue refineReducedInterval(const ExpressionValue &expr, const Interval &newInterval) {
  ExpressionValue refined = expr.withInterval(newInterval);
  if (expr.getInterval() != newInterval) {
    refined = refined.dropUnreducedInterval();
  }
  return refined;
}

bool isInMaybeSkippedScfRegion(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
    if (llvm::isa<FuncDefOp>(parent)) {
      return false;
    }

    // `writeResults` is a storage side-channel, not a path-sensitive lattice.
    // Writes nested under branch/loop control may not be absolute on every path through the
    // enclosing op, so keep the prior state instead of treating the nested write as unconditional.
    if (llvm::isa<
            scf::ForOp, scf::IfOp, scf::WhileOp, scf::IndexSwitchOp, scf::ParallelOp,
            scf::ForallOp>(parent)) {
      return true;
    }
  }
  return false;
}

std::optional<UnreducedInterval> getBooleanUnreducedInterval(const Interval &interval) {
  return interval.isBoolean() ? std::optional<UnreducedInterval>(interval.firstUnreduced())
                              : std::nullopt;
}

FailureOr<std::vector<SourceRef>>
translateRef(const SourceRef &ref, const SourceRefRemappings &translations) {
  std::vector<SourceRef> refs;
  for (const auto &[prefix, vals] : translations) {
    if (!ref.isValidPrefix(prefix)) {
      continue;
    }

    if (vals.isArray()) {
      auto suffix = ref.getSuffix(prefix);
      ensure(succeeded(suffix), "prefix checked before SourceRef suffix extraction");

      std::vector<SourceRefIndex> arraySuffix, remainingSuffix;
      bool suffixIsPastArray = false;
      for (const SourceRefIndex &idx : *suffix) {
        if (!suffixIsPastArray && arraySuffix.size() < vals.getNumArrayDims() &&
            (idx.isIndex() || idx.isIndexRange())) {
          arraySuffix.push_back(idx);
          continue;
        }
        suffixIsPastArray = true;
        remainingSuffix.push_back(idx);
      }

      auto resolvedValsRes = vals.extract(arraySuffix);
      ensure(succeeded(resolvedValsRes), "could not resolve translated SourceRef array child");
      SourceRefSet folded = resolvedValsRes->first.foldToScalar();
      if (remainingSuffix.empty()) {
        refs.insert(refs.end(), folded.begin(), folded.end());
        continue;
      }

      for (const SourceRef &baseRef : folded) {
        auto translatedRef = mlir::FailureOr<SourceRef>(baseRef);
        for (const SourceRefIndex &idx : remainingSuffix) {
          if (failed(translatedRef)) {
            break;
          }
          translatedRef = translatedRef->createChild(idx);
        }
        if (succeeded(translatedRef)) {
          refs.push_back(*translatedRef);
        }
      }
    } else {
      for (const SourceRef &replacement : vals.getScalarValue()) {
        auto translated = ref.translate(prefix, replacement);
        if (succeeded(translated)) {
          refs.push_back(*translated);
        }
      }
    }
  }

  if (refs.empty()) {
    return failure();
  }
  return refs;
}

bool isDirectSourceRefValue(Value value) {
  if (llvm::isa<BlockArgument>(value)) {
    return true;
  }

  Operation *definingOp = value.getDefiningOp();
  return llvm::isa_and_present<MemberReadOp, ReadArrayOp, polymorphic::ConstReadOp>(definingOp);
}

std::optional<SourceRefLatticeValue>
getIdentitySourceRefState(DataFlowSolver &solver, Value value) {
  if (isDirectSourceRefValue(value)) {
    SourceRefLatticeValue val = SourceRefAnalysis::getValueState(solver, value);
    if (val.isScalar()) {
      return val;
    }
    return std::nullopt;
  }

  auto createArray = llvm::dyn_cast_if_present<CreateArrayOp>(value.getDefiningOp());
  if (!createArray) {
    return std::nullopt;
  }

  SourceRefLatticeValue arrayVal(createArray.getType().getShape());
  for (auto [idx, element] : llvm::enumerate(createArray.getElements())) {
    std::optional<SourceRefLatticeValue> elementVal = getIdentitySourceRefState(solver, element);
    if (!elementVal.has_value()) {
      return std::nullopt;
    }
    (void)arrayVal.getElemFlatIdx(idx).setValue(*elementVal);
  }
  return arrayVal;
}

llvm::EquivalenceClasses<SourceRef>
collectDirectEqualityRefs(DataFlowSolver &solver, FuncDefOp fn) {
  llvm::EquivalenceClasses<SourceRef> eqRefs;
  fn.walk([&](EmitEqualityOp eqOp) {
    Operation *op = eqOp.getOperation();
    if (!dataflow::isOperationLive(solver, op)) {
      return;
    }

    Value lhs = eqOp.getLhs();
    Value rhs = eqOp.getRhs();
    if (!isDirectSourceRefValue(lhs) || !isDirectSourceRefValue(rhs)) {
      return;
    }

    SourceRefLatticeValue lhsState = SourceRefAnalysis::getValueState(solver, lhs);
    SourceRefLatticeValue rhsState = SourceRefAnalysis::getValueState(solver, rhs);
    if (!lhsState.isScalar() || !rhsState.isScalar() || !lhsState.isSingleValue() ||
        !rhsState.isSingleValue()) {
      return;
    }

    const SourceRef &lhsRef = lhsState.getSingleValue();
    const SourceRef &rhsRef = rhsState.getSingleValue();
    if (lhsRef.isConstant() || rhsRef.isConstant()) {
      return;
    }
    eqRefs.unionSets(lhsRef, rhsRef);
  });
  return eqRefs;
}

} // namespace

/* ExpressionValue */

/// Build the inverse relation using a symbol reserved by the calling analysis.
llvm::SMTExprRef createFieldInverseExpr(
    const llvm::SMTSolverRef &solver, const ExpressionValue &val, llvm::SMTExprRef invSym
) {
  const Field &field = val.getField();
  const Interval &iv = val.getInterval();
  if (iv.isDegenerate() && iv.lhs() != field.zero()) {
    DynamicAPInt invVal = field.inv(iv.lhs());
    return solver->mkBitvector(toAPSInt(invVal), field.bitWidth());
  }

  // The definition of an inverse X^-1 is Y s.t. XY % prime = 1.
  // Use the operation's cached symbol for Y and add the XY % prime = 1 constraint.
  llvm::SMTExprRef one = solver->mkBitvector(APSInt::get(1), field.bitWidth());
  llvm::SMTExprRef prime = solver->mkBitvector(toAPSInt(field.prime()), field.bitWidth());
  llvm::SMTExprRef mult = solver->mkBVMul(val.getExpr(), invSym);
  llvm::SMTExprRef mod = solver->mkBVURem(mult, prime);
  llvm::SMTExprRef constraint = solver->mkEqual(mod, one);
  solver->addConstraint(constraint);
  return invSym;
}

bool ExpressionValue::operator==(const ExpressionValue &rhs) const {
  if (expr == nullptr && rhs.expr == nullptr) {
    return i == rhs.i && unreduced == rhs.unreduced;
  }
  if (expr == nullptr || rhs.expr == nullptr) {
    return false;
  }
  return i == rhs.i && unreduced == rhs.unreduced && *expr == *rhs.expr;
}

ExpressionValue
boolToFelt(const llvm::SMTSolverRef &solver, const ExpressionValue &expr, unsigned bitwidth) {
  llvm::SMTExprRef zero = solver->mkBitvector(mlir::APSInt::get(0), bitwidth);
  llvm::SMTExprRef one = solver->mkBitvector(mlir::APSInt::get(1), bitwidth);
  llvm::SMTExprRef boolToFeltConv = solver->mkIte(expr.getExpr(), one, zero);
  return expr.withExpression(boolToFeltConv);
}

ExpressionValue selectValue(
    const llvm::SMTSolverRef &solver, const ExpressionValue &cond, const ExpressionValue &trueVal,
    const ExpressionValue &falseVal
) {
  const Field &f = trueVal.getField();
  const Interval &condInterval = cond.getInterval();
  Interval resultInterval;
  if (condInterval.isEmpty()) {
    resultInterval = Interval::Empty(f);
  } else if (condInterval.isDegenerate() && condInterval.rhs() == f.one()) {
    resultInterval = trueVal.getInterval();
  } else if (condInterval.isDegenerate() && condInterval.rhs() == f.zero()) {
    resultInterval = falseVal.getInterval();
  } else {
    resultInterval = trueVal.getInterval().join(falseVal.getInterval());
  }
  llvm::SMTExprRef resultExpr =
      solver->mkIte(cond.getExpr(), trueVal.getExpr(), falseVal.getExpr());
  std::optional<UnreducedInterval> resultUnreduced;
  if (condInterval.isEmpty()) {
    resultUnreduced = resultInterval.firstUnreduced();
  } else if (condInterval.isDegenerate() && condInterval.rhs() == f.one()) {
    resultUnreduced = trueVal.getOptionalUnreducedInterval();
  } else if (condInterval.isDegenerate() && condInterval.rhs() == f.zero()) {
    resultUnreduced = falseVal.getOptionalUnreducedInterval();
  } else {
    resultUnreduced = mergeUnreducedIntervals(
        trueVal.getOptionalUnreducedInterval(), falseVal.getOptionalUnreducedInterval()
    );
  }
  return ExpressionValue(resultExpr, resultInterval, std::move(resultUnreduced));
}

ExpressionValue intersection(
    const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs
) {
  Interval res = lhs.i.intersect(rhs.i);
  const auto *exprEq = solver->mkEqual(lhs.expr, rhs.expr);
  return ExpressionValue(exprEq, res);
}

ExpressionValue
add(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i + rhs.i;
  res.expr = solver->mkBVAdd(lhs.expr, rhs.expr);
  res = res.withOptionalUnreducedInterval(combineUnreducedIntervals(lhs, rhs, std::plus {}));
  return res;
}

ExpressionValue
sub(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i - rhs.i;
  res.expr = solver->mkBVSub(lhs.expr, rhs.expr);
  res = res.withOptionalUnreducedInterval(combineUnreducedIntervals(lhs, rhs, std::minus {}));
  return res;
}

ExpressionValue
mul(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i * rhs.i;
  res.expr = solver->mkBVMul(lhs.expr, rhs.expr);
  res = res.withOptionalUnreducedInterval(combineUnreducedIntervals(lhs, rhs, std::multiplies {}));
  return res;
}

ExpressionValue
div(const llvm::SMTSolverRef &solver, Operation *op, const ExpressionValue &lhs,
    const ExpressionValue &rhs, llvm::SMTExprRef inverseSymbol) {
  ExpressionValue res;
  auto divRes = feltDiv(lhs.i, rhs.i);
  if (failed(divRes)) {
    const Field &field = lhs.getField();
    const Interval &rhsInterval = rhs.getInterval();
    Interval zero = Interval::Degenerate(field, field.zero());
    if (!rhsInterval.isDegenerate()) {
      if (rhsInterval.intersect(zero).isNotEmpty()) {
        op->emitWarning(
              "non-degenerate felt.div divisors are not tracked precisely, and the divisor may "
              "contain zero. Range of division result will be treated as unbounded."
        )
            .report();
      } else {
        op->emitWarning(
              "non-degenerate felt.div divisors are not tracked precisely because precise field "
              "division over intervals would require enumerating divisor inverses. Range of "
              "division result will be treated as unbounded."
        )
            .report();
      }
    } else {
      op->emitWarning(
            "divisor is zero, leading to a divide-by-zero error. Range of division result will "
            "be treated as unbounded."
      )
          .report();
    }
    res.i = Interval::Entire(lhs.getField());
  } else {
    res.i = *divRes;
  }
  llvm::SMTExprRef invExpr = createFieldInverseExpr(solver, rhs, inverseSymbol);
  res.expr = solver->mkBVMul(lhs.expr, invExpr);
  return res;
}

ExpressionValue uintDiv(
    const llvm::SMTSolverRef &solver, Operation *op, const ExpressionValue &lhs,
    const ExpressionValue &rhs
) {
  ExpressionValue res;
  auto divRes = unsignedIntDiv(lhs.i, rhs.i);
  if (failed(divRes)) {
    op->emitWarning(
          "divisor is not restricted to non-zero values, leading to potential divide-by-zero error."
          " Range of division result will be treated as unbounded."
    )
        .report();
    res.i = Interval::Entire(lhs.getField());
  } else {
    res.i = *divRes;
  }
  res.expr = solver->mkBVUDiv(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue sintDiv(
    const llvm::SMTSolverRef &solver, Operation *op, const ExpressionValue &lhs,
    const ExpressionValue &rhs
) {
  ExpressionValue res;
  auto divRes = signedIntDiv(lhs.i, rhs.i);
  if (failed(divRes)) {
    op->emitWarning(
          "divisor is not restricted to non-zero values, leading to potential divide-by-zero error."
          " Range of division result will be treated as unbounded."
    )
        .report();
    res.i = Interval::Entire(lhs.getField());
  } else {
    res.i = *divRes;
  }
  res.expr = solver->mkBVSDiv(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
mod(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i % rhs.i;
  res.expr = solver->mkBVURem(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
sintMod(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  return ExpressionValue(
      solver->mkBVSRem(lhs.getExpr(), rhs.getExpr()),
      signedMod(lhs.getInterval(), rhs.getInterval())
  );
}

ExpressionValue
bitAnd(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i & rhs.i;
  res.expr = solver->mkBVAnd(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
bitOr(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = lhs.i | rhs.i;
  res.expr = solver->mkBVOr(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
bitXor(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  if (lhs.isBoolSort(solver) && rhs.isBoolSort(solver)) {
    return boolXor(solver, lhs, rhs);
  }

  ExpressionValue res;
  res.i = lhs.i ^ rhs.i;
  res.expr = solver->mkBVXor(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue shiftLeft(
    const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs
) {
  ExpressionValue res;
  res.i = lhs.i << rhs.i;
  res.expr = solver->mkBVShl(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue shiftRight(
    const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs
) {
  ExpressionValue res;
  res.i = lhs.i >> rhs.i;
  res.expr = solver->mkBVLshr(lhs.expr, rhs.expr);
  return res;
}

ExpressionValue
cmp(const llvm::SMTSolverRef &solver, CmpOp op, const ExpressionValue &lhs,
    const ExpressionValue &rhs) {
  ExpressionValue res;
  const Field &f = lhs.getField();
  // Default result is any boolean output for when we are unsure about the comparison result.
  res.i = Interval::Boolean(f);
  switch (op.getPredicate()) {
  case FeltCmpPredicate::EQ:
    res.expr = solver->mkEqual(lhs.expr, rhs.expr);
    if (lhs.i.isDegenerate() && rhs.i.isDegenerate()) {
      res.i = lhs.i == rhs.i ? Interval::True(f) : Interval::False(f);
    } else if (lhs.i.intersect(rhs.i).isEmpty()) {
      res.i = Interval::False(f);
    }
    break;
  case FeltCmpPredicate::NE:
    res.expr = solver->mkNot(solver->mkEqual(lhs.expr, rhs.expr));
    if (lhs.i.isDegenerate() && rhs.i.isDegenerate()) {
      res.i = lhs.i != rhs.i ? Interval::True(f) : Interval::False(f);
    } else if (lhs.i.intersect(rhs.i).isEmpty()) {
      res.i = Interval::True(f);
    }
    break;
  case FeltCmpPredicate::LT:
    res.expr = solver->mkBVUlt(lhs.expr, rhs.expr);
    if (lhs.i.toUnreduced().computeGEPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::True(f);
    }
    if (lhs.i.toUnreduced().computeLTPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::False(f);
    }
    break;
  case FeltCmpPredicate::LE:
    res.expr = solver->mkBVUle(lhs.expr, rhs.expr);
    if (lhs.i.toUnreduced().computeGTPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::True(f);
    }
    if (lhs.i.toUnreduced().computeLEPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::False(f);
    }
    break;
  case FeltCmpPredicate::GT:
    res.expr = solver->mkBVUgt(lhs.expr, rhs.expr);
    if (lhs.i.toUnreduced().computeLEPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::True(f);
    }
    if (lhs.i.toUnreduced().computeGTPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::False(f);
    }
    break;
  case FeltCmpPredicate::GE:
    res.expr = solver->mkBVUge(lhs.expr, rhs.expr);
    if (lhs.i.toUnreduced().computeLTPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::True(f);
    }
    if (lhs.i.toUnreduced().computeGEPart(rhs.i.toUnreduced()).reduce(f).isEmpty()) {
      res.i = Interval::False(f);
    }
    break;
  }
  res = res.withOptionalUnreducedInterval(getBooleanUnreducedInterval(res.i));
  return res;
}

ExpressionValue
boolAnd(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = boolAnd(lhs.i, rhs.i);
  res.expr = solver->mkAnd(lhs.expr, rhs.expr);
  res = res.withOptionalUnreducedInterval(getBooleanUnreducedInterval(res.i));
  return res;
}

ExpressionValue
boolOr(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = boolOr(lhs.i, rhs.i);
  res.expr = solver->mkOr(lhs.expr, rhs.expr);
  res = res.withOptionalUnreducedInterval(getBooleanUnreducedInterval(res.i));
  return res;
}

ExpressionValue
boolXor(const llvm::SMTSolverRef &solver, const ExpressionValue &lhs, const ExpressionValue &rhs) {
  ExpressionValue res;
  res.i = boolXor(lhs.i, rhs.i);
  // There's no Xor, so we do (L || R) && !(L && R)
  res.expr = solver->mkAnd(
      solver->mkOr(lhs.expr, rhs.expr), solver->mkNot(solver->mkAnd(lhs.expr, rhs.expr))
  );
  res = res.withOptionalUnreducedInterval(getBooleanUnreducedInterval(res.i));
  return res;
}

ExpressionValue neg(const llvm::SMTSolverRef &solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = -val.i;
  res.expr = solver->mkBVNeg(val.expr);
  if (val.hasUnreducedInterval()) {
    res = res.withOptionalUnreducedInterval(-val.getUnreducedInterval());
  }
  return res;
}

ExpressionValue notOp(const llvm::SMTSolverRef &solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = ~val.i;
  res.expr = solver->mkBVNot(val.expr);
  return res;
}

ExpressionValue boolNot(const llvm::SMTSolverRef &solver, const ExpressionValue &val) {
  ExpressionValue res;
  res.i = boolNot(val.i);
  res.expr = solver->mkNot(val.expr);
  res = res.withOptionalUnreducedInterval(getBooleanUnreducedInterval(res.i));
  return res;
}

ExpressionValue fallbackUnaryOp(
    const llvm::SMTSolverRef &solver, Operation *op, const ExpressionValue &val,
    llvm::SMTExprRef inverseSymbol
) {
  const Field &field = val.getField();
  ExpressionValue res;
  res.i = Interval::Entire(field);
  res.expr = TypeSwitch<Operation *, llvm::SMTExprRef>(op)
                 .Case<InvFeltOp>([&](auto) {
    return createFieldInverseExpr(solver, val, inverseSymbol);
  }).Default([](Operation *unsupported) {
    llvm::report_fatal_error(
        "no fallback provided for " + mlir::Twine(unsupported->getName().getStringRef())
    );
    return nullptr;
  });

  if (llvm::isa<InvFeltOp>(op)) {
    // We have the inverse's unreduced range to be [0, p-1] because for any integer z we can always
    // choose a conical element x \in [0, p-1] such that 1) (z * x) %p = 0 if z = 0, 2) (z * x) % p
    // = 1
    res = res.withOptionalUnreducedInterval(UnreducedInterval(field.zero(), field.maxVal()));
  }

  return res;
}

void ExpressionValue::print(mlir::raw_ostream &os) const {
  if (expr) {
    expr->print(os);
  } else {
    os << "<null expression>";
  }

  os << " ( interval: " << i << " )";
  if (unreduced.has_value()) {
    os << " ( unreduced: " << *unreduced << " )";
  }
}

/* IntervalAnalysisLattice */

ChangeResult IntervalAnalysisLattice::join(const AbstractSparseLattice & /*other*/) {
  // The update logic is handled in visitOperation; we don't support a generic
  // join operation, as it may override valid intervals.
  return ChangeResult::NoChange;
}

ChangeResult IntervalAnalysisLattice::meet(const AbstractSparseLattice & /*other*/) {
  // The update logic is handled in visitOperation; we don't support a generic
  // meet operation, as it may override valid intervals.
  return ChangeResult::NoChange;
}

void IntervalAnalysisLattice::print(mlir::raw_ostream &os) const {
  os << "IntervalAnalysisLattice { " << val << " }";
}

ChangeResult IntervalAnalysisLattice::setValue(const LatticeValue &newVal) {
  if (val == newVal) {
    return ChangeResult::NoChange;
  }
  val = newVal;
  return ChangeResult::Change;
}

ChangeResult IntervalAnalysisLattice::setValue(const ExpressionValue &e) {
  LatticeValue newVal(e);
  return setValue(newVal);
}

ChangeResult IntervalAnalysisLattice::addSolverConstraint(const ExpressionValue &e) {
  if (!constraints.contains(e)) {
    constraints.insert(e);
    return ChangeResult::Change;
  }
  return ChangeResult::NoChange;
}

/* IntervalDataFlowAnalysis */

IntervalDataFlowAnalysis::IntervalDataFlowAnalysis(
    mlir::DataFlowSolver &dataflowSolver, llvm::SMTSolverRef smt, const Field &f,
    bool propInputConstraints, bool shouldTrackUnreducedIntervals
)
    : Base::SparseForwardDataFlowAnalysis(dataflowSolver), _dataflowSolver(dataflowSolver),
      smtSolver(std::move(smt)), symbolNamespace(allocateSymbolNamespace(smtSolver)), field(f),
      propagateInputConstraints(propInputConstraints),
      trackUnreducedIntervals(shouldTrackUnreducedIntervals) {}

SourceRefLatticeValue IntervalDataFlowAnalysis::getSourceRefState(Value val) {
  // Components copy handles, so constraints must retain their original storage identity.
  if (llvm::isa<StructType>(val.getType())) {
    return SourceRefAnalysis::getValueState(_dataflowSolver, val);
  }
  Operation *definition = val.getDefiningOp();
  if (llvm::isa_and_present<CreateArrayOp>(definition)) {
    // Interval writes live at this result root; dependency sets describe initializer inputs.
    return SourceRefLatticeValue(SourceRef(llvm::cast<OpResult>(val)));
  }
  // Scalar and aggregate reads must use the same materialized roots as their writes.
  if (auto readPod = llvm::dyn_cast_if_present<pod::ReadPodOp>(definition)) {
    auto refs =
        getSourceRefState(readPod.getPodRef()).referencePodRecord(readPod.getRecordNameAttr());
    ensure(succeeded(refs), "could not resolve pod read");
    return refs->first;
  }
  if (auto readArray = llvm::dyn_cast_if_present<ReadArrayOp>(definition)) {
    auto refs = getSourceRefState(readArray.getArrRef())
                    .extract(getArrayAccessIndices(definition, readArray));
    ensure(succeeded(refs), "could not resolve array read");
    return refs->first;
  }
  if (auto readMember = llvm::dyn_cast_if_present<MemberReadOp>(definition)) {
    auto member = readMember.getMemberDefOp(tables);
    ensure(succeeded(member), "could not resolve member definition");
    auto refs = getSourceRefState(readMember.getComponent()).referenceMember(*member);
    ensure(succeeded(refs), "could not resolve member read");
    return refs->first;
  }
  return SourceRefAnalysis::getValueState(_dataflowSolver, val);
}

std::vector<SourceRefIndex> IntervalDataFlowAnalysis::getArrayAccessIndices(
    Operation * /*baseOp*/, ArrayAccessOpInterface arrayAccessOp
) {
  std::vector<SourceRefIndex> indices;
  ArrayType arrayType = arrayAccessOp.getArrRefType();
  size_t numIndices = arrayAccessOp.getIndices().size();
  indices.reserve(numIndices);

  for (size_t i = 0; i < numIndices; ++i) {
    Value idxOperand = arrayAccessOp.getIndices()[i];
    SourceRefLatticeValue idxVals = getSourceRefState(idxOperand);

    // Only exact constant indices get tracked precisely.
    auto constantIndex = idxVals.getConstantValue(field.get());
    if (succeeded(constantIndex)) {
      indices.emplace_back(*constantIndex);
    } else {
      indices.push_back(SourceRefIndex::forArrayDimension(arrayType.getDimSize(i)));
    }
  }

  return indices;
}

mlir::FailureOr<SourceRef> IntervalDataFlowAnalysis::getArrayAccessRef(
    Operation *baseOp, ArrayAccessOpInterface arrayAccessOp
) {
  std::vector<SourceRefIndex> indices = getArrayAccessIndices(baseOp, arrayAccessOp);
  Value arrayVal = arrayAccessOp.getArrRef();
  if (auto blockArg = llvm::dyn_cast<BlockArgument>(arrayVal)) {
    return SourceRef(blockArg, std::move(indices));
  }
  if (auto result = llvm::dyn_cast<OpResult>(arrayVal)) {
    return SourceRef(result, std::move(indices));
  }
  return failure();
}

Interval IntervalDataFlowAnalysis::getRefInterval(const SourceRef &ref) {
  if (auto it = writeResults.find(ref); it != writeResults.end()) {
    return it->second.getInterval();
  }

  if (ref.isConstantInt()) {
    auto constVal = ref.getConstantValue(field.get());
    if (succeeded(constVal)) {
      return Interval::Degenerate(field.get(), *constVal);
    }
  }

  if (ref.isRooted() && ref.getPath().empty()) {
    auto rootVal = ref.getRoot();
    if (succeeded(rootVal) && !llvm::isa<ArrayType, StructType, pod::PodType>(rootVal->getType())) {
      const ExpressionValue &rootExpr = getLatticeElement(*rootVal)->getValue().getScalarValue();
      if (rootExpr.getExpr() != nullptr) {
        return rootExpr.getInterval();
      }
    }
  }

  return getDefaultIntervalForType(ref.getType());
}

std::optional<UnreducedInterval>
IntervalDataFlowAnalysis::getDefaultUnreducedIntervalForType(mlir::Type ty) const {
  if (!trackUnreducedIntervals) {
    return std::nullopt;
  }
  if (isBooleanType(ty)) {
    return UnreducedInterval(0, 1);
  }
  return UnreducedInterval(field.get().zero(), field.get().maxVal());
}

std::optional<UnreducedInterval>
IntervalDataFlowAnalysis::getRefUnreducedInterval(const SourceRef &ref) {
  if (!trackUnreducedIntervals) {
    return std::nullopt;
  }

  if (auto it = writeResults.find(ref); it != writeResults.end()) {
    return it->second.getOptionalUnreducedInterval();
  }

  if (ref.isConstantInt()) {
    auto constVal = ref.getConstantValue(field.get());
    if (succeeded(constVal)) {
      return UnreducedInterval(*constVal, *constVal);
    }
  }

  if (ref.isRooted() && ref.getPath().empty()) {
    auto rootVal = ref.getRoot();
    if (succeeded(rootVal) && !llvm::isa<ArrayType, StructType, pod::PodType>(rootVal->getType())) {
      const ExpressionValue &rootExpr = getLatticeElement(*rootVal)->getValue().getScalarValue();
      if (rootExpr.hasUnreducedInterval()) {
        return rootExpr.getUnreducedInterval();
      }
    }
  }

  return getRefInterval(ref).firstUnreduced();
}

ExpressionValue
IntervalDataFlowAnalysis::getRefValue(const SourceRef &ref, llvm::SMTExprRef selectionSymbol) {
  const bool rangedSelection = hasRangedSelection(ref);
  std::optional<ExpressionValue> result;
  // Exact ranged entries also need coverage checks before discarding their no-write default.
  if (auto it = writeResults.find(ref); it != writeResults.end() && !rangedSelection) {
    result = it->second;
  } else {
    // A bounded selection needs no default if every concrete element has a stored value.
    // Count existing entries instead of enumerating potentially huge array dimensions.
    llvm::DynamicAPInt required(1);
    for (const SourceRefIndex &index : ref.getPath()) {
      if (index.isIndexRange()) {
        if (index.hasUnboundedUpperBound()) {
          required = -1;
          break;
        }
        auto [low, high] = index.getIndexRange();
        required *= high - low;
      }
    }
    auto concreteWrites = llvm::count_if(writeResults, [&](const auto &entry) {
      const SourceRef &written = entry.first;
      return written.getPath().size() == ref.getPath().size() &&
             hasOverlappingPrefix(written, ref) &&
             llvm::none_of(written.getPath(), [](const SourceRefIndex &index) {
        return index.isIndexRange();
      });
    });
    if (concreteWrites == 0 || required != llvm::DynamicAPInt(concreteWrites)) {
      result = ExpressionValue(selectionSymbol, getRefInterval(ref), getRefUnreducedInterval(ref));
    }
  }
  auto joinWrite = [&](const ExpressionValue &writtenVal) {
    if (!result) {
      result = writtenVal;
      return;
    }
    const auto *expr = result->getExpr();
    if (!expr || !writtenVal.getExpr() || *expr != *writtenVal.getExpr()) {
      expr = getOrCreateSymbol(ref);
    }
    result = ExpressionValue(
        expr, result->getInterval().join(writtenVal.getInterval()),
        mergeUnreducedIntervals(
            result->getOptionalUnreducedInterval(), writtenVal.getOptionalUnreducedInterval()
        )
    );
  };
  for (const auto &[writtenRef, writtenVal] : writeResults) {
    if ((writtenRef == ref && !rangedSelection) ||
        writtenRef.getPath().size() != ref.getPath().size() ||
        !hasOverlappingPrefix(writtenRef, ref)) {
      continue;
    }
    // Concrete entries retain their own no-write alternatives. Ranged entries contribute
    // only actual writes; uncovered selections already received a fallback above.
    if (!hasRangedSelection(writtenRef)) {
      joinWrite(writtenVal);
    } else {
      for (const auto &[writer, contribution] : writeContributions.at(writtenRef)) {
        joinWrite(contribution.value);
      }
    }
  }
  ensure(result.has_value(), "covered storage selection must have a value");
  // Equal ranges describe possible locations, not equality of independently selected values.
  return rangedSelection ? result->withExpression(selectionSymbol) : *result;
}

SmallVector<std::pair<SourceRef, ExpressionValue>> IntervalDataFlowAnalysis::getAggregateWrites(
    const SourceRef &source, const SourceRef &destination
) {
  SourceRefSet selections;
  for (const auto &[writtenRef, writtenVal] : writeResults) {
    if (!hasOverlappingPrefix(writtenRef, source)) {
      continue;
    }
    SourceRef selected = source;
    for (const SourceRefIndex &index : writtenRef.getPath().drop_front(source.getPath().size())) {
      auto child = selected.createChild(index);
      ensure(succeeded(child), "could not select aggregate descendant");
      selected = *child;
    }
    selections.insert(selected);
  }
  SmallVector<std::pair<SourceRef, ExpressionValue>> writes;
  for (const SourceRef &selected : selections) {
    auto translated = selected.translate(source, destination);
    ensure(succeeded(translated), "could not translate aggregate descendant");
    // Resolve the complete selection once for each destination leaf.
    writes.emplace_back(*translated, getRefValue(selected, getOrCreateSymbol(*translated)));
  }
  return writes;
}

void IntervalDataFlowAnalysis::recordRefWrite(
    const SourceRef &writtenRef, const ExpressionValue &writeVal, bool mayBeSkipped,
    Operation *publisher
) {
  std::optional<ExpressionValue> previous;
  if (auto it = writeResults.find(writtenRef); it != writeResults.end()) {
    previous = it->second;
  }
  auto joinStoredWrite = [this, &writtenRef](
                             const ExpressionValue &old, const ExpressionValue &next
                         ) -> ExpressionValue {
    Interval combinedWrite = old.getInterval().join(next.getInterval());
    auto combinedUnreduced = mergeUnreducedIntervals(
        old.getOptionalUnreducedInterval(), next.getOptionalUnreducedInterval()
    );
    if (old.getExpr() != nullptr && next.getExpr() != nullptr &&
        *old.getExpr() == *next.getExpr()) {
      return old.withInterval(combinedWrite).withOptionalUnreducedInterval(combinedUnreduced);
    }

    return ExpressionValue(
        getOrCreateSymbol(writtenRef), combinedWrite, std::move(combinedUnreduced)
    );
  };

  auto &contributions = writeContributions[writtenRef];
  if (auto it = contributions.find(publisher); !publisher && it != contributions.end()) {
    it->second.value = joinStoredWrite(it->second.value, writeVal);
    it->second.mayBeSkipped &= mayBeSkipped;
  } else {
    contributions.insert_or_assign(publisher, WriteContribution {writeVal, mayBeSkipped});
  }

  // A prior publication is not another possible write. Recompute from current contributions,
  // retaining the no-write alternative only while every contribution may be skipped.
  std::optional<ExpressionValue> combined;
  if (llvm::all_of(contributions, [](const auto &entry) { return entry.second.mayBeSkipped; })) {
    combined = ExpressionValue(
        getOrCreateSymbol(writtenRef), getDefaultIntervalForType(writtenRef.getType()),
        getDefaultUnreducedIntervalForType(writtenRef.getType())
    );
  }
  for (const auto &[writer, contribution] : contributions) {
    combined = combined ? joinStoredWrite(*combined, contribution.value) : contribution.value;
  }
  writeResults[writtenRef] = *combined;

  const ExpressionValue &readerUpdate =
      mayBeSkipped || publisher ? writeResults[writtenRef] : writeVal;
  for (Lattice *readerLattice : readResults[writtenRef]) {
    ExpressionValue prior = readerLattice->getValue().getScalarValue();
    Interval intersection = prior.getInterval().intersect(readerUpdate.getInterval());
    ExpressionValue newVal = prior.withInterval(intersection);
    propagateIfChanged(readerLattice, readerLattice->setValue(newVal));
  }

  if (!previous || *previous != writeResults.at(writtenRef)) {
    // Overlap handles concrete writes to ranged subscribers and the reverse direction.
    // Prefix subscriptions also catch newly discovered fields in a previously empty copy.
    for (const auto &[source, consumers] : aggregateReaders) {
      if (hasOverlappingPrefix(writtenRef, source)) {
        for (Operation *consumer : consumers) {
          _dataflowSolver.enqueue({getProgramPointAfter(consumer), this});
        }
      }
    }
  }
}

mlir::LogicalResult IntervalDataFlowAnalysis::visitOperation(
    Operation *op, ArrayRef<const Lattice *> operands, ArrayRef<Lattice *> results
) {
  // We only perform the visitation on operations within functions
  FuncDefOp fn = op->getParentOfType<FuncDefOp>();
  if (!fn) {
    return success();
  }

  // If there are no operands or results, skip.
  if (operands.empty() && results.empty()) {
    return success();
  }

  // Get the values or defaults from the operand lattices
  llvm::SmallVector<LatticeValue> operandVals;
  llvm::SmallVector<std::optional<SourceRef>> operandRefs;
  auto resolveRefStateValue =
      [&](Value value, const SourceRefLatticeValue &refSet) -> std::optional<LatticeValue> {
    ensure(refSet.isScalar(), "should have ruled out array values already");

    if (refSet.getScalarValue().empty()) {
      // If we can't compute the reference, then there must be some unsupported
      // op the reference analysis cannot handle. We emit a warning and return
      // early, since there's no meaningful computation we can do for this op.
      op->emitWarning()
          .append(
              "state of ", value,
              " is empty; defining operation is unsupported by SourceRef analysis"
          )
          .report();
      return std::nullopt;
    }

    if (!refSet.isSingleValue()) {
      Interval joinedInterval = Interval::Empty(field.get());
      std::optional<UnreducedInterval> joinedUnreduced = std::nullopt;
      bool sawFirst = false;
      for (const SourceRef &ref : refSet.getScalarValue()) {
        auto refValue = getRefValue(ref, getOrCreateSymbol(value));
        joinedInterval = joinedInterval.join(refValue.getInterval());
        const auto &refUnreduced = refValue.getOptionalUnreducedInterval();
        if (!sawFirst) {
          joinedUnreduced = refUnreduced;
          sawFirst = true;
        } else {
          joinedUnreduced = mergeUnreducedIntervals(joinedUnreduced, refUnreduced);
        }
      }
      ExpressionValue anyVal = createUnknownValue(value)
                                   .withInterval(joinedInterval)
                                   .withOptionalUnreducedInterval(joinedUnreduced);
      return LatticeValue(anyVal);
    }

    return LatticeValue(getRefValue(refSet.getSingleValue(), getOrCreateSymbol(value)));
  };
  for (unsigned opNum = 0; opNum < op->getNumOperands(); ++opNum) {
    Value val = op->getOperand(opNum);
    SourceRefLatticeValue refSet = getSourceRefState(val);
    if (llvm::isa<ArrayType, StructType, pod::PodType>(val.getType())) {
      // Aggregate storage changes independently of its SSA lattice. Observe prefixes even
      // when their descendants have not been written yet.
      for (const SourceRef &source : refSet.foldToScalar()) {
        aggregateReaders[source].insert(op);
      }
    }
    if (refSet.isSingleValue()) {
      operandRefs.push_back(refSet.getSingleValue());
    } else {
      operandRefs.push_back(std::nullopt);
    }
    // First, lookup the operand value after it is initialized
    auto priorState = operands[opNum]->getValue();
    if (priorState.getScalarValue().getExpr() != nullptr) {
      operandVals.push_back(priorState);
      continue;
    }

    if (auto readArr = llvm::dyn_cast_if_present<ReadArrayOp>(val.getDefiningOp())) {
      auto arrayRef = getArrayAccessRef(op, readArr);
      if (succeeded(arrayRef)) {
        if (auto it = writeResults.find(*arrayRef); it != writeResults.end()) {
          auto stored = getRefValue(*arrayRef, getOrCreateSymbol(val));
          operandVals.emplace_back(stored);
          Lattice *operandLattice = getLatticeElement(val);
          (void)operandLattice->setValue(stored);
          continue;
        }
      }
    }

    // Else, look up the stored value by `SourceRef`.
    // We only care about scalar type values here, so ignore aggregate storage values.
    Type valTy = val.getType();
    if (llvm::isa<ArrayType, StructType, pod::PodType>(valTy)) {
      ExpressionValue anyVal(field.get(), getOrCreateSymbol(val));
      operandVals.emplace_back(anyVal);
      continue;
    }

    std::optional<LatticeValue> resolvedValue;
    if (auto argument = llvm::dyn_cast<BlockArgument>(val)) {
      resolvedValue = LatticeValue(getRefValue(SourceRef(argument), getOrCreateSymbol(val)));
    } else {
      Operation *definition = val.getDefiningOp();
      if (isReadOp(definition) || isConstOp(definition)) {
        resolvedValue = resolveRefStateValue(val, refSet);
      } else {
        // Dependency sets erase the operation that produced a value. In particular,
        // a call returning `a + a` cannot inherit the exact value or interval of `a`.
        // Without an interval expression, keep an independent unknown SSA value.
        resolvedValue = LatticeValue(createUnknownValue(val));
      }
    }
    if (!resolvedValue.has_value()) {
      // We still return success so we can return overapproximated and partial
      // results to the user.
      return success();
    }
    operandVals.push_back(*resolvedValue);

    // Since we initialized a value that was not found in the before lattice,
    // update that value in the lattice so we can find it later, but we don't
    // need to propagate the changes, since we already have what we need.
    Lattice *operandLattice = getLatticeElement(val);
    (void)operandLattice->setValue(operandVals[opNum]);
  }

  if (isReadOp(op) && op->getNumResults() == 1) {
    Value resultVal = op->getResult(0);
    if (!llvm::isa<ArrayType, StructType, pod::PodType>(resultVal.getType())) {
      auto resolvedValue = resolveRefStateValue(resultVal, getSourceRefState(resultVal));
      if (resolvedValue.has_value()) {
        propagateIfChanged(results[0], results[0]->setValue(*resolvedValue));
      }
    }
    return success();
  }

  // Now, the way we update is dependent on the type of the operation.
  if (isConstOp(op)) {
    llvm::DynamicAPInt constVal = getConst(op);
    llvm::SMTExprRef expr;
    if (isBoolConstOp(op)) {
      expr = createConstBoolExpr(constVal != 0);
    } else {
      expr = createConstBitvectorExpr(constVal);
    }

    ExpressionValue latticeVal(field.get(), expr, constVal);
    if (trackUnreducedIntervals) {
      latticeVal = latticeVal.withUnreducedInterval(UnreducedInterval(constVal, constVal));
    }
    propagateIfChanged(results[0], results[0]->setValue(latticeVal));
  } else if (isArithmeticOp(op)) {
    ExpressionValue result;
    if (operands.size() == 2) {
      result = performBinaryArithmetic(op, operandVals[0], operandVals[1]);
    } else {
      result = performUnaryArithmetic(op, operandVals[0]);
    }

    // Also intersect with prior interval, if it's initialized
    const ExpressionValue &prior = results[0]->getValue().getScalarValue();
    if (prior.getExpr()) {
      result = refineReducedInterval(result, result.getInterval().intersect(prior.getInterval()));
    }
    propagateIfChanged(results[0], results[0]->setValue(result));
  } else if (auto selectOp = llvm::dyn_cast<arith::SelectOp>(op)) {
    ExpressionValue result = selectValue(
        smtSolver, operandVals[0].getScalarValue(), operandVals[1].getScalarValue(),
        operandVals[2].getScalarValue()
    );
    const ExpressionValue &prior = results[0]->getValue().getScalarValue();
    if (prior.getExpr()) {
      result = refineReducedInterval(result, result.getInterval().intersect(prior.getInterval()));
    }
    propagateIfChanged(results[0], results[0]->setValue(result));
  } else if (EmitEqualityOp emitEq = llvm::dyn_cast<EmitEqualityOp>(op)) {
    Value lhsVal = emitEq.getLhs(), rhsVal = emitEq.getRhs();
    ExpressionValue lhsExpr = operandVals[0].getScalarValue();
    ExpressionValue rhsExpr = operandVals[1].getScalarValue();

    // Special handling for generalized (s - c0) * (s - c1) * ... * (s - cN) = 0 patterns.
    // These patterns enforce that s is one of c0, ..., cN.
    auto res = getGeneralizedDecompInterval(op, lhsVal, rhsVal);
    if (succeeded(res)) {
      for (Value signalVal : res->first) {
        applyInterval(emitEq, signalVal, res->second);
      }
    }

    ExpressionValue constraint = intersection(smtSolver, lhsExpr, rhsExpr);
    // Update the LHS and RHS to the same value, but restricted intervals
    // based on the constraints.
    const Interval &constrainInterval = constraint.getInterval();
    applyInterval(emitEq, lhsVal, constrainInterval);
    applyInterval(emitEq, rhsVal, constrainInterval);
  } else if (auto assertOp = llvm::dyn_cast<AssertOp>(op)) {
    // assert enforces that the operand is true. So we apply an interval of [1, 1]
    // to the operand.
    Value cond = assertOp.getCondition();
    applyInterval(assertOp, cond, Interval::True(field.get()));
    // Also add the solver constraint that the expression must be true.
    auto assertExpr = operandVals[0].getScalarValue();
    // No need to propagate the constraint
    (void)getLatticeElement(cond)->addSolverConstraint(assertExpr);
  } else if (auto writePod = llvm::dyn_cast<pod::WritePodOp>(op)) {
    SourceRefLatticeValue podRefs = getSourceRefState(writePod.getPodRef());
    if (podRefs.isScalar()) {
      auto recordRefsRes = podRefs.referencePodRecord(writePod.getRecordNameAttr());
      ensure(succeeded(recordRefsRes), "could not create SourceRef child for pod write");
      SourceRefLatticeValue recordRefs = recordRefsRes->first;
      // A write to one of several possible PODs can miss each candidate.
      const bool maySkipWrite = isInMaybeSkippedScfRegion(op) || !recordRefs.isSingleValue();
      Type valueTy = writePod.getValue().getType();
      if (!llvm::isa<ArrayType, StructType, pod::PodType>(valueTy)) {
        ExpressionValue writeVal = operandVals[1].getScalarValue();
        for (const SourceRef &recordRef : recordRefs.getScalarValue()) {
          recordRefWrite(recordRef, writeVal, maySkipWrite);
        }
      } else if (operandRefs[1].has_value()) {
        llvm::SmallVector<std::pair<SourceRef, ExpressionValue>> remappedWrites;
        for (const SourceRef &recordRef : recordRefs.getScalarValue()) {
          llvm::append_range(remappedWrites, getAggregateWrites(*operandRefs[1], recordRef));
        }
        for (const auto &[translatedRef, translatedVal] : remappedWrites) {
          recordRefWrite(translatedRef, translatedVal, maySkipWrite, /*publisher=*/op);
        }
      }
    }
  } else if (auto writem = llvm::dyn_cast<MemberWriteOp>(op)) {
    const bool maySkipWrite = isInMaybeSkippedScfRegion(op);
    // Update values stored in a member
    ExpressionValue writeVal = operandVals[1].getScalarValue();
    auto cmp = writem.getComponent();
    // We also need to update the interval on the assigned symbol
    SourceRefLatticeValue refSet = getSourceRefState(cmp);
    if (refSet.isSingleValue()) {
      auto memberDefRes = writem.getMemberDefOp(tables);
      if (succeeded(memberDefRes)) {
        SourceRefIndex idx(memberDefRes.value());
        auto memberRefRes = refSet.getSingleValue().createChild(idx);
        ensure(succeeded(memberRefRes), "could not create SourceRef child for member write");
        const SourceRef &memberRef = *memberRefRes;
        Type memberTy = writem.getVal().getType();
        if (!llvm::isa<ArrayType, StructType, pod::PodType>(memberTy)) {
          // Simple scalar update
          recordRefWrite(memberRef, writeVal, maySkipWrite);
        } else {
          // Map the intervals of aggregates to the written member
          std::optional<SourceRef> rhsPrefix;
          if (operandRefs[1].has_value() && operandRefs[1]->isRooted()) {
            rhsPrefix = operandRefs[1];
          } else if (auto blockArg = llvm::dyn_cast<BlockArgument>(writem.getVal())) {
            rhsPrefix = SourceRef(blockArg);
          } else if (auto result = llvm::dyn_cast<OpResult>(writem.getVal())) {
            rhsPrefix = SourceRef(result);
          }

          if (rhsPrefix.has_value()) {
            for (const auto &[translatedRef, translatedVal] :
                 getAggregateWrites(*rhsPrefix, memberRef)) {
              recordRefWrite(translatedRef, translatedVal, maySkipWrite, /*publisher=*/op);
            }
          }
        }
      }
    }
  } else if (auto writeArr = llvm::dyn_cast<WriteArrayOp>(op)) {
    ExpressionValue writeVal = operandVals.back().getScalarValue();
    auto arrayRef = getArrayAccessRef(op, writeArr);
    // A ranged access may miss each candidate even when the operation always executes.
    const bool maySkipWrite =
        isInMaybeSkippedScfRegion(op) || (succeeded(arrayRef) && hasRangedSelection(*arrayRef));
    if (succeeded(arrayRef)) {
      recordRefWrite(*arrayRef, writeVal, maySkipWrite);
    }

    SourceRefLatticeValue arrayVals = getSourceRefState(writeArr.getArrRef());
    if (arrayVals.isScalar()) {
      std::vector<SourceRefIndex> indices = getArrayAccessIndices(op, writeArr);
      auto targetRefsRes = arrayVals.extract(indices);
      ensure(succeeded(targetRefsRes), "could not create SourceRef child for array write");
      auto [targetRefs, _] = *targetRefsRes;
      ensure(targetRefs.isScalar(), "array write must resolve to scalar references");
      for (const SourceRef &ref : targetRefs.getScalarValue()) {
        recordRefWrite(
            ref, writeVal, maySkipWrite || !targetRefs.isSingleValue() || hasRangedSelection(ref)
        );
      }
    }
  } else if (auto createArray = llvm::dyn_cast<CreateArrayOp>(op)) {
    const auto &elements = createArray.getElements();
    ArrayType arrayTy = createArray.getType();
    Type elemTy = arrayTy.getElementType();

    if (!elements.empty() && !llvm::isa<ArrayType, StructType, pod::PodType>(elemTy)) {
      ensure(arrayTy.hasStaticShape(), "array.new with explicit elements must have static shape");
      ensure(
          std::cmp_equal(elements.size(), arrayTy.getNumElements()),
          "array.new explicit initializer length must match array shape"
      );

      array::ArrayIndexGen indexGen = array::ArrayIndexGen::from(arrayTy);
      OpResult arrayRes = createArray->getResult(0);
      for (unsigned i = 0; i < elements.size(); ++i) {
        auto maybeIndices = indexGen.delinearize(i, op->getContext());
        ensure(maybeIndices.has_value(), "could not delinearize array.new element index");

        SourceRef::Path path;
        path.reserve(maybeIndices->size());
        for (Attribute attr : *maybeIndices) {
          auto idxAttr = llvm::dyn_cast<IntegerAttr>(attr);
          ensure(idxAttr != nullptr, "array.new delinearize should produce integer attributes");
          path.emplace_back(integerAttrToDynamicAPInt(idxAttr));
        }

        recordRefWrite(
            SourceRef(arrayRes, std::move(path)), operandVals[i].getScalarValue(),
            /*mayBeSkipped=*/false, /*publisher=*/op
        );
      }
    } else if (!elements.empty()) {
      ensure(arrayTy.hasStaticShape(), "aggregate array.new initializer requires static shape");
      array::ArrayIndexGen indexGen = array::ArrayIndexGen::from(arrayTy);
      SourceRef arrayRoot(llvm::cast<OpResult>(createArray.getResult()));
      llvm::SmallVector<std::pair<SourceRef, ExpressionValue>> remappedWrites;
      for (auto [i, element] : llvm::enumerate(elements)) {
        SourceRefLatticeValue elementRefs = getSourceRefState(element);
        if (!elementRefs.isSingleValue()) {
          continue;
        }
        auto maybeIndices = indexGen.delinearize(checkedCast<int64_t>(i), op->getContext());
        ensure(maybeIndices.has_value(), "could not delinearize aggregate array.new index");
        SourceRef elementTarget = arrayRoot;
        for (Attribute attr : *maybeIndices) {
          auto child = elementTarget.createChild(
              SourceRefIndex(integerAttrToDynamicAPInt(llvm::cast<IntegerAttr>(attr)))
          );
          ensure(succeeded(child), "could not create aggregate array element SourceRef");
          elementTarget = *child;
        }
        llvm::append_range(
            remappedWrites, getAggregateWrites(elementRefs.getSingleValue(), elementTarget)
        );
      }
      for (const auto &[translatedRef, translatedVal] : remappedWrites) {
        recordRefWrite(translatedRef, translatedVal, /*mayBeSkipped=*/false, /*publisher=*/op);
      }
    }
  } else if (auto newPod = llvm::dyn_cast<pod::NewPodOp>(op)) {
    SourceRef podRoot(llvm::cast<OpResult>(newPod.getResult()));
    for (auto [idx, record] : llvm::enumerate(newPod.getInitializedRecordValues())) {
      auto recordRef =
          podRoot.createChild(SourceRefIndex(StringAttr::get(op->getContext(), record.name)));
      ensure(succeeded(recordRef), "could not create SourceRef child for pod initializer");
      if (!llvm::isa<ArrayType, StructType, pod::PodType>(record.value.getType())) {
        recordRefWrite(
            *recordRef, operandVals[idx].getScalarValue(), /*mayBeSkipped=*/false,
            /*publisher=*/op
        );
        continue;
      }

      SourceRefLatticeValue sourceRefs = getSourceRefState(record.value);
      if (!sourceRefs.isSingleValue()) {
        continue;
      }
      for (const auto &[translatedRef, translatedVal] :
           getAggregateWrites(sourceRefs.getSingleValue(), *recordRef)) {
        recordRefWrite(translatedRef, translatedVal, /*mayBeSkipped=*/false, /*publisher=*/op);
      }
    }
  } else if (
      auto intToFelt = dyn_cast<IntToFeltOp>(op);
      intToFelt && intToFelt.getOverflow() != cast::OverflowSemantics::ASSERT &&
      !intToFelt.getValue().getType().isInteger(1)
  ) {
    // Non-assert casts can change the value non-monotonically (other than boolean inputs which
    // cannot overflow and retain their relation in the cast case below). Forget the operand
    // relation, but initialize the result so sparse dataflow visits its users.
    propagateIfChanged(results[0], results[0]->setValue(createUnknownValue(intToFelt.getResult())));
  } else if (
      auto feltToIndex = dyn_cast<FeltToIndexOp>(op);
      feltToIndex && feltToIndex.getOverflow() != cast::OverflowSemantics::ASSERT
  ) {
    // See the IntToFeltOp case above.
    propagateIfChanged(results[0], results[0]->setValue(createUnknownValue(op->getResult(0))));
  } else if (isa<IntToFeltOp, FeltToIndexOp>(op)) {
    // Casts don't modify the intervals, but they do modify the SMT types.
    ExpressionValue expr = operandVals[0].getScalarValue();
    // We treat all ints and indexes as felts with the exception of comparison
    // results, which are bools. So if `expr` is a bool, this cast needs to
    // upcast to a felt.
    if (expr.isBoolSort(smtSolver)) {
      expr = boolToFelt(smtSolver, expr, field.get().bitWidth());
    }
    propagateIfChanged(results[0], results[0]->setValue(expr));
  } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
    Operation *parent = op->getParentOp();
    ensure(parent, "yield operation must have parent operation");

    // Bind scalar yielded values to the corresponding parent results.
    for (unsigned idx = 0; idx < yieldOp.getResults().size(); ++idx) {
      Value parentRes = parent->getResult(idx);
      Lattice *resLattice = getLatticeElement(parentRes);
      // Merge with the existing value, if present (e.g., another branch)
      // has possible value that must be merged.
      ExpressionValue exprVal = resLattice->getValue().getScalarValue();
      ExpressionValue newResVal = operandVals[idx].getScalarValue();
      if (auto loopOp = llvm::dyn_cast<LoopLikeOpInterface>(parent)) {
        // We overapproximate for loops because we aren't going to try to track trip count.
        newResVal = ExpressionValue(getOrCreateSymbol(parentRes), Interval::Entire(field.get()));
      }
      if (exprVal.getExpr() != nullptr) {
        newResVal =
            exprVal.withInterval(exprVal.getInterval().join(newResVal.getInterval()))
                .withOptionalUnreducedInterval(mergeUnreducedIntervals(
                    exprVal.getOptionalUnreducedInterval(), newResVal.getOptionalUnreducedInterval()
                ));
      } else {
        newResVal = ExpressionValue(
            getOrCreateSymbol(parentRes), newResVal.getInterval(),
            newResVal.getOptionalUnreducedInterval()
        );
      }
      propagateIfChanged(resLattice, resLattice->setValue(newResVal));
    }

    // Aggregate intervals are tracked in `writeResults`, so remap writes
    // rooted at each yielded aggregate onto the corresponding result.
    for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {
      Value parentRes = parent->getResult(idx);
      if (!llvm::isa<ArrayType, StructType, pod::PodType>(parentRes.getType())) {
        continue;
      }

      SourceRefLatticeValue yieldedRefs = getSourceRefState(yieldOp.getOperand(idx));
      if (!yieldedRefs.isScalar()) {
        continue;
      }

      SourceRef resultRef(llvm::cast<OpResult>(parentRes));
      llvm::MapVector<SourceRef, ExpressionValue> remappedWrites;
      // Join all possible sources before replacing this yield's publication for each leaf.
      for (const SourceRef &yieldedRef : yieldedRefs.getScalarValue()) {
        for (const auto &[translatedRef, translatedVal] :
             getAggregateWrites(yieldedRef, resultRef)) {
          auto [it, inserted] = remappedWrites.try_emplace(translatedRef, translatedVal);
          if (inserted) {
            continue;
          }
          const ExpressionValue &previous = it->second;
          const auto *expr = previous.getExpr();
          if (!expr || !translatedVal.getExpr() || *expr != *translatedVal.getExpr()) {
            expr = getOrCreateSymbol(translatedRef);
          }
          it->second = ExpressionValue(
              expr, previous.getInterval().join(translatedVal.getInterval()),
              mergeUnreducedIntervals(
                  previous.getOptionalUnreducedInterval(),
                  translatedVal.getOptionalUnreducedInterval()
              )
          );
        }
      }
      // Branch yields remain independent publishers, even when they share a parent result.
      for (const auto &[translatedRef, translatedVal] : remappedWrites) {
        recordRefWrite(translatedRef, translatedVal, /*mayBeSkipped=*/false, /*publisher=*/op);
      }
    }
  } else if (
      // We do not need to explicitly handle read ops since they are resolved at the operand value
      // step where `SourceRef`s are queries.
      !isReadOp(op)
      // We do not currently handle return ops as the analysis is currently limited to constrain
      // functions, which return no value.
      && !isReturnOp(op)
      // The analysis ignores definition ops.
      && !isDefinitionOp(op)
      // We do not need to analyze storage creation directly.
      && !llvm::isa<CreateArrayOp, CreateStructOp, pod::NewPodOp, NonDetOp>(op)
  ) {
    op->emitWarning("unhandled operation, analysis may be incomplete").report();
  }

  return success();
}

llvm::SMTExprRef IntervalDataFlowAnalysis::getOrCreateSymbol(const SourceRef &r) {
  auto it = refSymbols.find(r);
  if (it != refSymbols.end()) {
    return it->second;
  }
  std::string name = symbolNamespace + "ref." + std::to_string(refSymbols.size());
  llvm::SMTExprRef sym = createSymbol(r.getType(), name.c_str());
  refSymbols[r] = sym;
  return sym;
}

llvm::SMTExprRef IntervalDataFlowAnalysis::createSymbol(mlir::Type ty, const char *name) const {
  if (isBooleanType(ty)) {
    return smtSolver->mkSymbol(name, smtSolver->getBoolSort());
  }
  return field.get().createSymbol(smtSolver, name);
}

llvm::SMTExprRef IntervalDataFlowAnalysis::getOrCreateSymbol(Value val) {
  auto it = valueSymbols.find(val);
  if (it != valueSymbols.end()) {
    return it->second;
  }
  std::string name = symbolNamespace + "value." + std::to_string(valueSymbols.size());
  llvm::SMTExprRef sym = createSymbol(val.getType(), name.c_str());
  valueSymbols[val] = sym;
  return sym;
}

llvm::SMTExprRef IntervalDataFlowAnalysis::getOrCreateInverseSymbol(Operation *op) {
  auto it = inverseSymbols.find(op);
  if (it != inverseSymbols.end()) {
    return it->second;
  }
  std::string name = symbolNamespace + "inverse." + std::to_string(inverseSymbols.size());
  llvm::SMTExprRef sym = field.get().createSymbol(smtSolver, name.c_str());
  inverseSymbols[op] = sym;
  return sym;
}

llvm::DynamicAPInt IntervalDataFlowAnalysis::getConst(Operation *op) const {
  ensure(isConstOp(op), "op is not a const op");

  // NOTE: I think clang-format makes these hard to read by default
  // clang-format off
  llvm::DynamicAPInt fieldConst = TypeSwitch<Operation *, llvm::DynamicAPInt>(op)
      .Case<FeltConstantOp>([&](auto feltConst) {
        // Preserve felt.const's declared-field meaning before reduction in the analysis field.
        // Fieldless literals instead use the raw fallback, as in SourceRef::getConstantValue.
        return field.get().reduce(feltConst.getValue().getReducedValueOrRaw());
      })
      .Case<arith::ConstantIndexOp>([&](auto indexConst) {
        return DynamicAPInt(indexConst.value());
      })
      .Case<arith::ConstantIntOp>([&](auto intConst) {
        auto valAttr = dyn_cast<IntegerAttr>(intConst.getValue());
        ensure(valAttr != nullptr, "arith::ConstantIntOp must have an IntegerAttr as its value");
        return integerAttrToDynamicAPInt(valAttr);
      })
      .Default([](auto *illegalOp) {
        std::string err;
        debug::Appender(err) << "unhandled getConst case: " << *illegalOp;
        llvm::report_fatal_error(Twine(err));
        return llvm::DynamicAPInt();
      });
  // clang-format on
  return fieldConst;
}

ExpressionValue IntervalDataFlowAnalysis::performBinaryArithmetic(
    Operation *op, const LatticeValue &a, const LatticeValue &b
) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto lhs = a.getScalarValue(), rhs = b.getScalarValue();
  ensure(lhs.getExpr(), "cannot perform arithmetic over null lhs smt expr");
  ensure(rhs.getExpr(), "cannot perform arithmetic over null rhs smt expr");

  // clang-format off
  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<AddFeltOp>([&](auto) { return add(smtSolver, lhs, rhs); })
                 .Case<SubFeltOp>([&](auto) { return sub(smtSolver, lhs, rhs); })
                 .Case<MulFeltOp>([&](auto) { return mul(smtSolver, lhs, rhs); })
                 .Case<DivFeltOp>([&](auto) {
                   return div(smtSolver, op, lhs, rhs, getOrCreateInverseSymbol(op));
                 })
                 .Case<UnsignedIntDivFeltOp>([&](auto) {return uintDiv(smtSolver, op, lhs, rhs); })
                 .Case<SignedIntDivFeltOp>([&](auto) {return sintDiv(smtSolver, op, lhs, rhs); })
                 .Case<UnsignedModFeltOp>([&](auto) { return mod(smtSolver, lhs, rhs); })
                 .Case<SignedModFeltOp>([&](auto) { return sintMod(smtSolver, lhs, rhs); })
                 .Case<AndFeltOp>([&](auto) { return bitAnd(smtSolver, lhs, rhs); })
                 .Case<OrFeltOp>([&](auto) { return bitOr(smtSolver, lhs, rhs); })
                 .Case<XorFeltOp, arith::XOrIOp>([&](auto) { return bitXor(smtSolver, lhs, rhs); })
                 .Case<ShlFeltOp>([&](auto) { return shiftLeft(smtSolver, lhs, rhs); })
                 .Case<ShrFeltOp>([&](auto) { return shiftRight(smtSolver, lhs, rhs); })
                 .Case<CmpOp>([&](auto cmpOp) { return cmp(smtSolver, cmpOp, lhs, rhs); })
                 .Case<AndBoolOp>([&](auto) { return boolAnd(smtSolver, lhs, rhs); })
                 .Case<OrBoolOp>([&](auto) { return boolOr(smtSolver, lhs, rhs); })
                 .Case<XorBoolOp>([&](auto) { return boolXor(smtSolver, lhs, rhs); })
  .Default([&](auto *unsupported) {
    unsupported
        ->emitError(
            "unsupported binary arithmetic operation"
        )
        .report();
    return ExpressionValue();
  });
  // clang-format on

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

ExpressionValue
IntervalDataFlowAnalysis::performUnaryArithmetic(Operation *op, const LatticeValue &a) {
  ensure(isArithmeticOp(op), "is not arithmetic op");

  auto val = a.getScalarValue();
  ensure(val.getExpr(), "cannot perform arithmetic over null smt expr");

  auto res = TypeSwitch<Operation *, ExpressionValue>(op)
                 .Case<NegFeltOp>([&](auto) { return neg(smtSolver, val); })
                 .Case<NotFeltOp>([&](auto) { return notOp(smtSolver, val); })
                 .Case<NotBoolOp>([&](auto) { return boolNot(smtSolver, val); })
                 // The inverse op is currently overapproximated
                 .Case<InvFeltOp>([&](auto inv) {
    return fallbackUnaryOp(smtSolver, inv, val, getOrCreateInverseSymbol(inv));
  }).Default([&](auto *unsupported) {
    unsupported
        ->emitWarning(
            "unsupported unary arithmetic operation, defaulting to over-approximated interval"
        )
        .report();
    return fallbackUnaryOp(smtSolver, unsupported, val, nullptr);
  });

  ensure(res.getExpr(), "arithmetic produced null smt expr");
  return res;
}

void IntervalDataFlowAnalysis::applyInterval(Operation *valUser, Value val, Interval newInterval) {
  Lattice *valLattice = getLatticeElement(val);
  ExpressionValue oldLatticeVal = valLattice->getValue().getScalarValue();
  // Intersect with the current value to accumulate restrictions across constraints.
  Interval intersection = oldLatticeVal.getInterval().intersect(newInterval);
  ExpressionValue newLatticeVal = refineReducedInterval(oldLatticeVal, intersection);
  ChangeResult changed = valLattice->setValue(newLatticeVal);

  if (auto blockArg = llvm::dyn_cast<BlockArgument>(val)) {
    auto fnOp = dyn_cast<FuncDefOp>(blockArg.getOwner()->getParentOp());

    // Apply the interval from the constrain function inputs to the compute function inputs
    if (propagateInputConstraints && fnOp && fnOp.isStructConstrain() &&
        blockArg.getArgNumber() > 0 && !newInterval.isEntire()) {
      auto structOp = fnOp->getParentOfType<StructDefOp>();
      FuncDefOp computeFn = structOp.getComputeFuncOp();
      BlockArgument computeArg = computeFn.getArgument(blockArg.getArgNumber() - 1);
      Lattice *computeEntryLattice = getLatticeElement(computeArg);

      SourceRef ref(computeArg);
      ExpressionValue newArgVal(
          getOrCreateSymbol(ref), newInterval,
          trackUnreducedIntervals ? std::optional<UnreducedInterval>(newInterval.firstUnreduced())
                                  : std::nullopt
      );
      propagateIfChanged(computeEntryLattice, computeEntryLattice->setValue(newArgVal));
    }
  }

  // Now we descend into val's operands, if it has any.
  Operation *definingOp = val.getDefiningOp();
  if (!definingOp) {
    propagateIfChanged(valLattice, changed);
    return;
  }

  const Field &f = field.get();

  // This is a rules-based operation. If we have a rule for a given operation,
  // then we can make some kind of update, otherwise we leave the intervals
  // as is.
  // - First we'll define all the rules so the type switch can be less messy

  // cmp.<pred> restricts each side of the comparison if the result is known.
  auto cmpCase = [&](CmpOp cmpOp) {
    // Cmp output range is [0, 1], so in order to do something, we must have newInterval
    // either "true" (1) or "false" (0).
    // -- In the case of a contradictory circuit, however, the cmp result is allowed
    // to be empty.
    ensure(
        newInterval.isBoolean() || newInterval.isEmpty(),
        "new interval for CmpOp is not boolean or empty"
    );
    if (!newInterval.isDegenerate()) {
      // The comparison result is unknown, so we can't update the operand ranges
      return;
    }

    bool cmpTrue = newInterval.rhs() == f.one();

    Value lhs = cmpOp.getLhs(), rhs = cmpOp.getRhs();
    auto lhsLat = getLatticeElement(lhs), rhsLat = getLatticeElement(rhs);
    ExpressionValue lhsExpr = lhsLat->getValue().getScalarValue(),
                    rhsExpr = rhsLat->getValue().getScalarValue();

    Interval newLhsInterval, newRhsInterval;
    const Interval &lhsInterval = lhsExpr.getInterval();
    const Interval &rhsInterval = rhsExpr.getInterval();

    FeltCmpPredicate pred = cmpOp.getPredicate();
    // predicate cases
    auto eqCase = [&]() {
      return (pred == FeltCmpPredicate::EQ && cmpTrue) ||
             (pred == FeltCmpPredicate::NE && !cmpTrue);
    };
    auto neCase = [&]() {
      return (pred == FeltCmpPredicate::NE && cmpTrue) ||
             (pred == FeltCmpPredicate::EQ && !cmpTrue);
    };
    auto ltCase = [&]() {
      return (pred == FeltCmpPredicate::LT && cmpTrue) ||
             (pred == FeltCmpPredicate::GE && !cmpTrue);
    };
    auto leCase = [&]() {
      return (pred == FeltCmpPredicate::LE && cmpTrue) ||
             (pred == FeltCmpPredicate::GT && !cmpTrue);
    };
    auto gtCase = [&]() {
      return (pred == FeltCmpPredicate::GT && cmpTrue) ||
             (pred == FeltCmpPredicate::LE && !cmpTrue);
    };
    auto geCase = [&]() {
      return (pred == FeltCmpPredicate::GE && cmpTrue) ||
             (pred == FeltCmpPredicate::LT && !cmpTrue);
    };

    // new intervals based on case
    if (eqCase()) {
      newLhsInterval = newRhsInterval = lhsInterval.intersect(rhsInterval);
    } else if (neCase()) {
      if (lhsInterval.isDegenerate() && rhsInterval.isDegenerate() && lhsInterval == rhsInterval) {
        // In this case, we know lhs and rhs cannot satisfy this assertion, so they have
        // an empty value range.
        newLhsInterval = newRhsInterval = Interval::Empty(f);
      } else if (lhsInterval.isDegenerate()) {
        // rhs must not overlap with lhs
        newLhsInterval = lhsInterval;
        newRhsInterval = rhsInterval.difference(lhsInterval);
      } else if (rhsInterval.isDegenerate()) {
        // lhs must not overlap with rhs
        newLhsInterval = lhsInterval.difference(rhsInterval);
        newRhsInterval = rhsInterval;
      } else {
        // Leave unchanged
        newLhsInterval = lhsInterval;
        newRhsInterval = rhsInterval;
      }
    } else if (ltCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeLTPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeGEPart(lhsInterval.toUnreduced()).reduce(f);
    } else if (leCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeLEPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeGTPart(lhsInterval.toUnreduced()).reduce(f);
    } else if (gtCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeGTPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeLEPart(lhsInterval.toUnreduced()).reduce(f);
    } else if (geCase()) {
      newLhsInterval = lhsInterval.toUnreduced().computeGEPart(rhsInterval.toUnreduced()).reduce(f);
      newRhsInterval = rhsInterval.toUnreduced().computeLTPart(lhsInterval.toUnreduced()).reduce(f);
    } else {
      cmpOp->emitWarning("unhandled cmp predicate").report();
      return;
    }

    // Now we recurse to each operand
    applyInterval(cmpOp, lhs, newLhsInterval);
    applyInterval(cmpOp, rhs, newRhsInterval);
  };

  // Multiplication cases:
  // - If the result of a multiplication is non-zero, then both operands must be
  // non-zero.
  // - If one operand is a constant, we can propagate the new interval when multiplied
  // by the multiplicative inverse of the constant.
  auto mulCase = [&](MulFeltOp mulOp) {
    // We check for the constant case first.
    auto constCase = [&](FeltConstantOp constOperand, Value multiplicand) {
      DynamicAPInt constVal = getConst(constOperand);
      if (constVal == 0) {
        // There's no inverse for zero, so we do nothing.
        return;
      }
      Interval updatedInterval = newInterval * Interval::Degenerate(f, f.inv(constVal));
      applyInterval(mulOp, multiplicand, updatedInterval);
    };

    Value lhs = mulOp.getLhs(), rhs = mulOp.getRhs();

    auto lhsConstOp = dyn_cast_if_present<FeltConstantOp>(lhs.getDefiningOp());
    auto rhsConstOp = dyn_cast_if_present<FeltConstantOp>(rhs.getDefiningOp());
    // If both are consts, we don't need to do anything
    if (lhsConstOp && rhsConstOp) {
      return;
    } else if (lhsConstOp) {
      constCase(lhsConstOp, rhs);
      return;
    } else if (rhsConstOp) {
      constCase(rhsConstOp, lhs);
      return;
    }

    // Otherwise, try to propagate non-zero information.
    auto zeroInt = Interval::Degenerate(f, f.zero());
    if (newInterval.intersect(zeroInt).isNotEmpty()) {
      // The multiplication may be zero, so we can't reduce the operands to be non-zero
      return;
    }

    auto lhsLat = getLatticeElement(lhs), rhsLat = getLatticeElement(rhs);
    ExpressionValue lhsExpr = lhsLat->getValue().getScalarValue(),
                    rhsExpr = rhsLat->getValue().getScalarValue();
    Interval newLhsInterval = lhsExpr.getInterval().difference(zeroInt);
    Interval newRhsInterval = rhsExpr.getInterval().difference(zeroInt);
    applyInterval(mulOp, lhs, newLhsInterval);
    applyInterval(mulOp, rhs, newRhsInterval);
  };

  auto addCase = [&](AddFeltOp addOp) {
    Value lhs = addOp.getLhs(), rhs = addOp.getRhs();
    Lattice *lhsLat = getLatticeElement(lhs), *rhsLat = getLatticeElement(rhs);
    ExpressionValue lhsVal = lhsLat->getValue().getScalarValue();
    ExpressionValue rhsVal = rhsLat->getValue().getScalarValue();

    const Interval &currLhsInt = lhsVal.getInterval(), &currRhsInt = rhsVal.getInterval();

    Interval derivedLhsInt = newInterval - currRhsInt;
    Interval derivedRhsInt = newInterval - currLhsInt;

    Interval finalLhsInt = currLhsInt.intersect(derivedLhsInt);
    Interval finalRhsInt = currRhsInt.intersect(derivedRhsInt);

    applyInterval(addOp, lhs, finalLhsInt);
    applyInterval(addOp, rhs, finalRhsInt);
  };

  auto subCase = [&](SubFeltOp subOp) {
    Value lhs = subOp.getLhs(), rhs = subOp.getRhs();
    Lattice *lhsLat = getLatticeElement(lhs), *rhsLat = getLatticeElement(rhs);
    ExpressionValue lhsVal = lhsLat->getValue().getScalarValue();
    ExpressionValue rhsVal = rhsLat->getValue().getScalarValue();

    const Interval &currLhsInt = lhsVal.getInterval(), &currRhsInt = rhsVal.getInterval();

    Interval derivedLhsInt = newInterval + currRhsInt;
    Interval derivedRhsInt = currLhsInt - newInterval;

    Interval finalLhsInt = currLhsInt.intersect(derivedLhsInt);
    Interval finalRhsInt = currRhsInt.intersect(derivedRhsInt);

    applyInterval(subOp, lhs, finalLhsInt);
    applyInterval(subOp, rhs, finalRhsInt);
  };

  auto selectCase = [&](arith::SelectOp selectOp) {
    Value cond = selectOp.getCondition();
    Value trueVal = selectOp.getTrueValue();
    Value falseVal = selectOp.getFalseValue();

    ExpressionValue condExpr = getLatticeElement(cond)->getValue().getScalarValue();
    ExpressionValue trueExpr = getLatticeElement(trueVal)->getValue().getScalarValue();
    ExpressionValue falseExpr = getLatticeElement(falseVal)->getValue().getScalarValue();

    const Interval &condInterval = condExpr.getInterval();
    if (condInterval.isDegenerate() && condInterval.rhs() == f.one()) {
      applyInterval(selectOp, trueVal, newInterval);
      return;
    }
    if (condInterval.isDegenerate() && condInterval.rhs() == f.zero()) {
      applyInterval(selectOp, falseVal, newInterval);
      return;
    }

    Interval trueOverlap = trueExpr.getInterval().intersect(newInterval);
    Interval falseOverlap = falseExpr.getInterval().intersect(newInterval);
    bool truePossible = trueOverlap.isNotEmpty();
    bool falsePossible = falseOverlap.isNotEmpty();

    if (truePossible && !falsePossible) {
      applyInterval(selectOp, cond, Interval::True(f));
      applyInterval(selectOp, trueVal, newInterval);
      return;
    }
    if (!truePossible && falsePossible) {
      applyInterval(selectOp, cond, Interval::False(f));
      applyInterval(selectOp, falseVal, newInterval);
      return;
    }
    if (!truePossible && !falsePossible) {
      applyInterval(selectOp, cond, Interval::Empty(f));
    }
  };

  auto propagateRangedRead = [&](const SourceRef &ref) {
    auto &readers = rangedReadResults[ref];
    readers.insert(valLattice);
    SmallVector<Lattice *> aliases;
    Interval shared = newLatticeVal.getInterval();
    for (Lattice *reader : readers) {
      if (haveSameReadValue(val, reader->getAnchor())) {
        aliases.push_back(reader);
        shared = shared.intersect(reader->getValue().getScalarValue().getInterval());
      }
    }
    for (Lattice *reader : aliases) {
      // Keep per-read symbols, but equate proven aliases for relational solver constraints.
      if (reader != valLattice) {
        smtSolver->addConstraint(smtSolver->mkEqual(
            newLatticeVal.getExpr(), reader->getValue().getScalarValue().getExpr()
        ));
      }
      ExpressionValue refined = refineReducedInterval(reader->getValue().getScalarValue(), shared);
      propagateIfChanged(reader, reader->setValue(refined));
    }
  };

  auto readmCase = [&](MemberReadOp) {
    SourceRefLatticeValue sourceRefVal = getSourceRefState(val);

    if (sourceRefVal.isSingleValue() && hasRangedSelection(sourceRefVal.getSingleValue())) {
      propagateRangedRead(sourceRefVal.getSingleValue());
    }
    if (sourceRefVal.isSingleValue() && !hasRangedSelection(sourceRefVal.getSingleValue())) {
      const SourceRef &ref = sourceRefVal.getSingleValue();
      readResults[ref].insert(valLattice);

      // Also propagate to all other member read results for this member
      for (Lattice *l : readResults[ref]) {
        if (l != valLattice) {
          propagateIfChanged(l, l->setValue(newLatticeVal));
        }
      }
    }
  };

  auto readArrCase = [&](ReadArrayOp) {
    auto arrayRef = getArrayAccessRef(valUser, llvm::cast<ReadArrayOp>(definingOp));
    if (succeeded(arrayRef) && hasRangedSelection(*arrayRef)) {
      propagateRangedRead(*arrayRef);
    }
    // A constraint on one selection cannot refine other reads of the same possible range.
    if (succeeded(arrayRef) && !hasRangedSelection(*arrayRef)) {
      readResults[*arrayRef].insert(valLattice);

      for (Lattice *l : readResults[*arrayRef]) {
        if (l != valLattice) {
          propagateIfChanged(l, l->setValue(newLatticeVal));
        }
      }
    }

    SourceRefLatticeValue sourceRefVal = getSourceRefState(val);

    if (sourceRefVal.isSingleValue() && !hasRangedSelection(sourceRefVal.getSingleValue())) {
      const SourceRef &ref = sourceRefVal.getSingleValue();
      readResults[ref].insert(valLattice);

      // Also propagate to all other member read results for this member
      for (Lattice *l : readResults[ref]) {
        if (l != valLattice) {
          propagateIfChanged(l, l->setValue(newLatticeVal));
        }
      }
    }
  };

  // Pass the interval to the operand when the cast preserves its value.
  // Boolean inputs cannot overflow in any IntToFeltOp mode.
  auto castCase = [&](Operation *op) {
    if (auto intToFelt = dyn_cast<IntToFeltOp>(op);
        intToFelt && intToFelt.getOverflow() != cast::OverflowSemantics::ASSERT &&
        !intToFelt.getValue().getType().isInteger(1)) {
      return;
    }
    if (auto feltToIndex = dyn_cast<FeltToIndexOp>(op);
        feltToIndex && feltToIndex.getOverflow() != cast::OverflowSemantics::ASSERT) {
      return;
    }
    applyInterval(op, op->getOperand(0), newInterval);
  };

  // - Apply the rules given the op.
  // NOTE: disabling clang-format for this because it makes the last case statement
  // look ugly.
  // clang-format off
  TypeSwitch<Operation *>(definingOp)
            .Case<CmpOp>([&](auto op) { cmpCase(op); })
            .Case<AddFeltOp>([&](auto op) { return addCase(op); })
            .Case<SubFeltOp>([&](auto op) { return subCase(op); })
            .Case<MulFeltOp>([&](auto op) { mulCase(op); })
            .Case<arith::SelectOp>([&](auto op) { selectCase(op); })
            .Case<MemberReadOp>([&](auto op){ readmCase(op); })
            .Case<ReadArrayOp>([&](auto op){ readArrCase(op); })
            .Case<IntToFeltOp, FeltToIndexOp>([&](auto op) { castCase(op); })
            .Default([&](Operation *) { });
  // clang-format on

  // Propagate after recursion to avoid having recursive calls unset the value.
  propagateIfChanged(valLattice, changed);
}

FailureOr<std::pair<DenseSet<Value>, Interval>>
IntervalDataFlowAnalysis::getGeneralizedDecompInterval(
    Operation * /*baseOp*/, Value lhs, Value rhs
) {
  auto isZeroConst = [this](Value v) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return false;
    }
    if (!isConstOp(op)) {
      return false;
    }
    return getConst(op) == field.get().zero();
  };
  bool lhsIsZero = isZeroConst(lhs), rhsIsZero = isZeroConst(rhs);
  Value exprTree = nullptr;
  if (lhsIsZero && !rhsIsZero) {
    exprTree = rhs;
  } else if (!lhsIsZero && rhsIsZero) {
    exprTree = lhs;
  } else {
    return failure();
  }

  // We now explore the expression tree for multiplications of subtractions/signal values.
  std::optional<SourceRef> signalRef = std::nullopt;
  DenseSet<Value> signalVals;
  SmallVector<DynamicAPInt> consts;
  SmallVector<Value> frontier {exprTree};
  while (!frontier.empty()) {
    Value v = frontier.back();
    frontier.pop_back();
    Operation *op = v.getDefiningOp();

    FeltConstantOp c;
    Value signalVal;
    auto handleRefValue = [this, &signalRef, &signalVal, &signalVals]() {
      SourceRefLatticeValue refSet = getSourceRefState(signalVal);
      if (!refSet.isScalar() || !refSet.isSingleValue()) {
        return failure();
      }
      SourceRef r = refSet.getSingleValue();
      // Matching ranges do not prove that independently selected values are equal.
      // Require SSA identity or a proven read alias for ranged selections.
      if (signalRef.has_value() &&
          (signalRef.value() != r ||
           (hasRangedSelection(r) && !haveSameReadValue(signalVal, *signalVals.begin())))) {
        return failure();
      } else if (!signalRef.has_value()) {
        signalRef = r;
      }
      signalVals.insert(signalVal);
      return success();
    };

    auto subPattern = m_CommutativeOp<SubFeltOp>(m_RefValue(&signalVal), m_Constant(&c));
    if (op && matchPattern(op, subPattern)) {
      if (failed(handleRefValue())) {
        return failure();
      }
      consts.push_back(getConst(c));
      continue;
    } else if (m_RefValue(&signalVal).match(v)) {
      if (failed(handleRefValue())) {
        return failure();
      }
      consts.push_back(field.get().zero());
      continue;
    }

    Value a, b;
    auto mulPattern = m_CommutativeOp<MulFeltOp>(matchers::m_Any(&a), matchers::m_Any(&b));
    if (op && matchPattern(op, mulPattern)) {
      frontier.push_back(a);
      frontier.push_back(b);
      continue;
    }

    return failure();
  }

  // Now, we aggregate the Interval. If we have sparse values (e.g., 0, 2, 4),
  // we will create a larger range of [0, 4], since we don't support multiple intervals.
  std::sort(consts.begin(), consts.end());
  Interval iv = UnreducedInterval(consts.front(), consts.back()).reduce(field.get());
  return std::make_pair(std::move(signalVals), iv);
}

/* StructIntervals */

LogicalResult StructIntervals::computeIntervals(
    mlir::DataFlowSolver &solver, mlir::AnalysisManager &am, const IntervalAnalysisContext &ctx
) {
  SymbolTableCollection tables;

  auto computeIntervalsImpl =
      [&solver, &am, &ctx, &tables, this](
          FuncDefOp fn, llvm::MapVector<SourceRef, Interval> &memberRanges,
          llvm::MapVector<SourceRef, UnreducedInterval> &memberUnreducedRanges,
          llvm::SetVector<ExpressionValue> & /*solverConstraints*/
      ) {
    auto setUnreducedRange =
        [&memberUnreducedRanges](const SourceRef &ref, const UnreducedInterval &interval) {
      memberUnreducedRanges.erase(ref);
      memberUnreducedRanges.insert({ref, interval});
    };
    // Since every lattice value does not contain every value, we will traverse
    // the function backwards (from most up-to-date to least-up-to-date lattices)
    // searching for the source refs. Once a source ref is found, we remove it
    // from the search set.

    SourceRefSet searchSet;
    for (const auto &ref : SourceRef::getAllSourceRefs(structDef, fn)) {
      // We only want to compute intervals for field elements and not composite types.
      if (!ref.isScalar()) {
        continue;
      }
      searchSet.insert(ref);
    }
    SourceRefSet functionRefs = searchSet;

    auto mergeInterval = [&memberRanges, &memberUnreducedRanges](
                             const SourceRef &ref, const Interval &interval,
                             std::optional<UnreducedInterval> unreducedInterval = std::nullopt
                         ) {
      auto *existing = memberRanges.find(ref);
      if (existing != memberRanges.end()) {
        Interval mergedInterval = existing->second.intersect(interval);
        bool intervalChanged = mergedInterval != existing->second;
        existing->second = std::move(mergedInterval);

        if (unreducedInterval.has_value()) {
          auto *existingUnreduced = memberUnreducedRanges.find(ref);
          if (existingUnreduced != memberUnreducedRanges.end()) {
            existingUnreduced->second = existingUnreduced->second.intersect(*unreducedInterval);
          } else {
            memberUnreducedRanges.insert({ref, *unreducedInterval});
          }
        } else if (intervalChanged) {
          memberUnreducedRanges.erase(ref);
        }
        return;
      }

      memberRanges[ref] = interval;
      if (unreducedInterval.has_value()) {
        memberUnreducedRanges.insert({ref, *unreducedInterval});
      }
    };

    // Iterate over arguments
    for (BlockArgument arg : fn.getArguments()) {
      SourceRef ref {arg};
      if (searchSet.erase(ref)) {
        const IntervalAnalysisLattice *lattice = solver.lookupState<IntervalAnalysisLattice>(arg);
        // If we never referenced this argument, use a default value
        ExpressionValue expr = lattice->getValue().getScalarValue();
        if (!expr.getExpr()) {
          expr = expr.withInterval(Interval::Entire(ctx.getField()));
          if (ctx.doTrackUnreducedIntervals()) {
            expr = expr.withUnreducedInterval(expr.getInterval().firstUnreduced());
          }
        }
        memberRanges[ref] = expr.getInterval();
        if (expr.hasUnreducedInterval()) {
          setUnreducedRange(ref, expr.getUnreducedInterval());
        }
        assert(memberRanges[ref].getField() == ctx.getField() && "bad interval defaults");
      }
    }

    // Aggregate all read intervals for a ref. A single ref may be read at multiple program
    // points with different precision, so picking an arbitrary lattice from the DenseSet is
    // nondeterministic. Joining preserves the overapproximation regardless of iteration order.
    for (const auto &[ref, lattices] : ctx.intervalDFA->getReadResults()) {
      if (!lattices.empty() && searchSet.erase(ref)) {
        Interval joinedInterval = Interval::Empty(ctx.getField());
        std::optional<UnreducedInterval> joinedUnreduced = std::nullopt;
        bool sawFirst = false;
        for (const IntervalAnalysisLattice *lattice : lattices) {
          const ExpressionValue &expr = lattice->getValue().getScalarValue();
          joinedInterval = joinedInterval.join(expr.getInterval());
          if (!sawFirst) {
            joinedUnreduced = expr.getOptionalUnreducedInterval();
            sawFirst = true;
          } else {
            joinedUnreduced =
                mergeUnreducedIntervals(joinedUnreduced, expr.getOptionalUnreducedInterval());
          }
        }
        memberRanges[ref] = joinedInterval;
        if (joinedUnreduced.has_value()) {
          setUnreducedRange(ref, *joinedUnreduced);
        }
        assert(memberRanges[ref].getField() == ctx.getField() && "bad interval defaults");
      }
    }

    for (const auto &[ref, val] : ctx.intervalDFA->getWriteResults()) {
      if (searchSet.erase(ref)) {
        memberRanges[ref] = val.getInterval();
        if (val.hasUnreducedInterval()) {
          setUnreducedRange(ref, val.getUnreducedInterval());
        }
        assert(memberRanges[ref].getField() == ctx.getField() && "bad interval defaults");
      }
    }

    // Child constrain calls refine parent-visible storage, but only after the callee
    // summary is translated through the call operands. If translation cannot prove
    // which parent ref owns a child interval, the local overapproximation remains.
    if (fn.isStructConstrain()) {
      auto mergeChildConstrainIntervals = [&](CallOp fnCall) {
        if (!dataflow::isOperationLive(solver, fnCall.getOperation())) {
          return;
        }

        auto res = resolveCallableSilently<FuncDefOp>(tables, fnCall);
        if (failed(res)) {
          return;
        }

        FuncDefOp calledFn = res->get();
        if (!calledFn.isStructConstrain()) {
          return;
        }

        auto calledStruct = calledFn->getParentOfType<StructDefOp>();
        if (calledStruct == structDef) {
          return;
        }

        auto &childAnalysis = am.getChildAnalysis<StructIntervalAnalysis>(calledStruct);
        if (childAnalysis.inProgress(ctx)) {
          return;
        }
        if (!childAnalysis.constructed(ctx)) {
          ensure(
              succeeded(childAnalysis.runAnalysis(solver, am, ctx)),
              "could not construct interval analysis for child struct"
          );
        }

        // Translate callee argument refs into parent refs and capture scalar call-site intervals
        // that can refine direct equality groups inside the child constrain function.
        SourceRefRemappings identityTranslations;
        llvm::MapVector<SourceRef, Interval> callOperandIntervals;
        for (unsigned i = 0; i < calledFn.getNumArguments(); i++) {
          SourceRef prefix(calledFn.getArgument(i));
          Value operand = fnCall.getOperand(i);
          std::optional<SourceRefLatticeValue> identityVal =
              getIdentitySourceRefState(solver, operand);
          if (identityVal.has_value()) {
            identityTranslations.emplace_back(prefix, *identityVal);
          }

          if (!llvm::isa<ArrayType, StructType, pod::PodType>(operand.getType())) {
            const IntervalAnalysisLattice *lattice =
                solver.lookupState<IntervalAnalysisLattice>(operand);
            if (lattice != nullptr) {
              const ExpressionValue &expr = lattice->getValue().getScalarValue();
              callOperandIntervals[prefix] = expr.getInterval();
            }
          }
        }

        const StructIntervals &childIntervals = childAnalysis.getResult(ctx);
        const auto &constrainIntervals = childIntervals.getConstrainIntervals();
        const auto &constrainUnreducedIntervals = childIntervals.getConstrainUnreducedIntervals();
        for (const auto &[childRef, childInterval] : constrainIntervals) {
          auto translatedRefs = translateRef(childRef, identityTranslations);
          if (failed(translatedRefs)) {
            continue;
          }

          std::optional<UnreducedInterval> childUnreduced = std::nullopt;
          if (const auto *childUnreducedIt = constrainUnreducedIntervals.find(childRef);
              childUnreducedIt != constrainUnreducedIntervals.end()) {
            childUnreduced = childUnreducedIt->second;
          }

          SourceRefSet uniqueTranslatedRefs;
          for (const SourceRef &translatedRef : *translatedRefs) {
            uniqueTranslatedRefs.insert(translatedRef);
          }
          if (uniqueTranslatedRefs.size() != 1) {
            continue;
          }

          const SourceRef &translatedRef = *uniqueTranslatedRefs.begin();
          if (functionRefs.contains(translatedRef)) {
            mergeInterval(translatedRef, childInterval, childUnreduced);
            searchSet.erase(translatedRef);
          }
        }

        // Direct equalities in the child can combine child summaries, call operand intervals, and
        // existing parent intervals before merging back into each translated parent ref.
        llvm::EquivalenceClasses<SourceRef> directEqRefs =
            collectDirectEqualityRefs(solver, calledFn);
        for (const auto *leader : directEqRefs) {
          if (!leader->isLeader()) {
            continue;
          }

          llvm::MapVector<SourceRef, Interval> translatedEqRefs;
          Interval contextualInterval = Interval::Entire(ctx.getField());
          bool hasInterval = false;
          bool ambiguousTranslation = false;

          for (const SourceRef &member : directEqRefs.members(*leader)) {
            Interval memberInterval = Interval::Entire(ctx.getField());
            if (const auto *childIntervalIt = constrainIntervals.find(member);
                childIntervalIt != constrainIntervals.end()) {
              memberInterval = memberInterval.intersect(childIntervalIt->second);
            }
            if (auto *callOperandIt = callOperandIntervals.find(member);
                callOperandIt != callOperandIntervals.end()) {
              memberInterval = memberInterval.intersect(callOperandIt->second);
              contextualInterval = contextualInterval.intersect(memberInterval);
              hasInterval = true;
            }

            auto translatedRefs = translateRef(member, identityTranslations);
            if (failed(translatedRefs)) {
              continue;
            }

            SourceRefSet uniqueTranslatedRefs;
            for (const SourceRef &translatedRef : *translatedRefs) {
              uniqueTranslatedRefs.insert(translatedRef);
            }
            if (uniqueTranslatedRefs.size() != 1) {
              ambiguousTranslation = true;
              break;
            }

            const SourceRef &translatedRef = *uniqueTranslatedRefs.begin();
            if (!functionRefs.contains(translatedRef)) {
              continue;
            }

            if (auto *parentIntervalIt = memberRanges.find(translatedRef);
                parentIntervalIt != memberRanges.end()) {
              memberInterval = memberInterval.intersect(parentIntervalIt->second);
            }

            translatedEqRefs[translatedRef] = memberInterval;
            contextualInterval = contextualInterval.intersect(memberInterval);
            hasInterval = true;
          }

          if (ambiguousTranslation || !hasInterval || translatedEqRefs.empty()) {
            continue;
          }

          for (const auto &[translatedRef, _] : translatedEqRefs) {
            mergeInterval(translatedRef, contextualInterval);
            searchSet.erase(translatedRef);
          }
        }
      };

      fn.walk(mergeChildConstrainIntervals);
    }

    // For all unfound refs, default to the entire range.
    for (const auto &ref : searchSet) {
      memberRanges[ref] = Interval::Entire(ctx.getField());
      if (ctx.doTrackUnreducedIntervals()) {
        setUnreducedRange(ref, memberRanges[ref].firstUnreduced());
      }
    }

    // Sort the outputs since we assembled things out of order.
    //
    // `llvm::MapVector` maintains an internal key -> index map. Sorting it in
    // place corrupts lookup semantics because the backing vector is reordered
    // without rebuilding that map. Reinsert into a fresh MapVector instead.
    llvm::SmallVector<std::pair<SourceRef, Interval>> sortedRanges;
    sortedRanges.reserve(memberRanges.size());
    for (const auto &[ref, interval] : memberRanges) {
      sortedRanges.emplace_back(ref, interval);
    }
    llvm::sort(sortedRanges, [](const auto &a, const auto &b) { return a.first < b.first; });
    llvm::SmallVector<std::pair<SourceRef, UnreducedInterval>> sortedUnreducedRanges;
    sortedUnreducedRanges.reserve(memberUnreducedRanges.size());
    for (const auto &[ref, interval] : memberUnreducedRanges) {
      sortedUnreducedRanges.emplace_back(ref, interval);
    }
    llvm::sort(sortedUnreducedRanges, [](const auto &a, const auto &b) {
      return a.first < b.first;
    });
    memberRanges.clear();
    memberUnreducedRanges.clear();
    for (auto &[ref, interval] : sortedRanges) {
      memberRanges[ref] = interval;
    }
    for (auto &[ref, interval] : sortedUnreducedRanges) {
      memberUnreducedRanges.insert({ref, interval});
    }
  };

  if (auto computeFn = structDef.getComputeFuncOp()) {
    computeIntervalsImpl(
        computeFn, computeMemberRanges, computeMemberUnreducedRanges, computeSolverConstraints
    );
  }
  if (auto constrainFn = structDef.getConstrainFuncOp()) {
    computeIntervalsImpl(
        constrainFn, constrainMemberRanges, constrainMemberUnreducedRanges,
        constrainSolverConstraints
    );
  }

  return success();
}

void StructIntervals::print(
    mlir::raw_ostream &os, bool withConstraints, bool printCompute, bool printUnreduced
) const {
  auto writeIntervals =
      [&os, &withConstraints, &printUnreduced](
          const char *fnName, const llvm::MapVector<SourceRef, Interval> &memberRanges,
          const llvm::MapVector<SourceRef, UnreducedInterval> &memberUnreducedRanges,
          const llvm::SetVector<ExpressionValue> &solverConstraints, bool printName
      ) {
    int indent = 4;
    if (printName) {
      os << '\n';
      os.indent(indent) << fnName << " {";
      indent += 4;
    }

    if (memberRanges.empty()) {
      os << "}\n";
      return;
    }

    for (const auto &[ref, interval] : memberRanges) {
      os << '\n';
      os.indent(indent) << ref << " in " << interval;
      if (printUnreduced) {
        const auto *unreducedIt = memberUnreducedRanges.find(ref);
        if (unreducedIt != memberUnreducedRanges.end()) {
          os << " ( " << unreducedIt->second << " )";
        }
      }
    }

    if (withConstraints) {
      os << "\n\n";
      os.indent(indent) << "Solver Constraints { ";
      if (solverConstraints.empty()) {
        os << "}\n";
      } else {
        for (const auto &e : solverConstraints) {
          os << '\n';
          os.indent(indent + 4);
          e.getExpr()->print(os);
        }
        os << '\n';
        os.indent(indent) << '}';
      }
    }

    if (printName) {
      os << '\n';
      os.indent(indent - 4) << '}';
    }
  };

  os << "StructIntervals { ";
  if (constrainMemberRanges.empty() && (!printCompute || computeMemberRanges.empty())) {
    os << "}\n";
    return;
  }

  if (printCompute) {
    writeIntervals(
        FUNC_NAME_COMPUTE, computeMemberRanges, computeMemberUnreducedRanges,
        computeSolverConstraints, printCompute
    );
  }
  writeIntervals(
      FUNC_NAME_CONSTRAIN, constrainMemberRanges, constrainMemberUnreducedRanges,
      constrainSolverConstraints, printCompute
  );

  os << "\n}\n";
}

} // namespace llzk
