//===-- Ops.cpp - PCL dialect implementation ----------------*- C++ -*-----===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "pcl/Dialect/IR/Ops.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "pcl/Dialect/IR/Attrs.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Matchers.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>

#include <llvm/Support/Debug.h>
#include <llvm/Support/LogicalResult.h>

#include <algorithm>

// TableGen'd implementation files
#define GET_OP_CLASSES
#include "pcl/Dialect/IR/Ops.cpp.inc"

using namespace pcl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

namespace {

/// Looks up the field's prime attribute in the module that contains the PCL circuit.
/// The op may not live inside a valid PCL module (i.e. during conversion). In that case
/// this function returns `std::nullopt`.
template <typename Op> std::optional<PrimeAttr> getFieldPrime(Op &op) {
  auto modOp = op->template getParentOfType<ModuleOp>();
  if (!modOp) {
    return std::nullopt;
  }

  auto attr = mlir::dyn_cast_if_present<PrimeAttr>(modOp->getAttr("pcl.prime"));
  if (!attr) {
    return std::nullopt;
  }

  return attr;
}

/// Used in the `isIdentity` and `isZero` callbacks for identifing if the queried value
/// is the LHS or the RHS of the operation.
enum class Side : std::uint8_t { Lhs, Rhs };

/// Folds a binary operation.
///
/// The helper is generic over the operation based on a set of callbacks:
/// - A callback that defines the actual operation
/// - A callback that queries if the given value is the identity under that operation (i.e. 1 for
/// multiplication or 0 for addition).
/// - A callback that queries if the given value "cancels out" the operation (i.e. 0 for
/// multiplication or false for conjunction).
/// - An optional callback that creates the fold result.
template <typename T, typename Op>
OpFoldResult foldBinaryOp(
    Op &op, typename Op::FoldAdaptor adaptor, llvm::function_ref<T(T, T)> opFn,
    llvm::function_ref<bool(T, Side)> isIdentity, llvm::function_ref<bool(T, Side)> isZero,
    llvm::function_ref<OpFoldResult(T)> factory = nullptr
) {
  auto factoryFn = [factory](auto value) -> OpFoldResult {
    if (factory) {
      return factory(value);
    }
    return value;
  };
  auto attrOrValue = [&factoryFn](auto attr, auto value) -> OpFoldResult {
    if (attr) {
      return factoryFn(attr);
    }
    return value;
  };

  auto lhs = mlir::dyn_cast_if_present<T>(adaptor.getLhs());
  auto rhs = mlir::dyn_cast_if_present<T>(adaptor.getRhs());
  // Shortcircuit if both operands are not constant.
  if (!rhs && !lhs) {
    return nullptr;
  }

  // If either side is "zero", then the operation is canceled out and return the "zero" attribute.
  if (lhs && isZero(lhs, Side::Lhs)) {
    return factoryFn(lhs);
  }
  if (rhs && isZero(rhs, Side::Rhs)) {
    return factoryFn(rhs);
  }
  // If either side is the identity, return the other side.
  // If the other side is a constant, return the attribute representing it.
  // Otherwise, return the value of the operand.
  if (lhs && isIdentity(lhs, Side::Lhs)) {
    return attrOrValue(rhs, op.getRhs());
  }
  if (rhs && isIdentity(rhs, Side::Rhs)) {
    return attrOrValue(lhs, op.getLhs());
  }
  // If both are constants but none matched the identity or "zero" predicates, perform the
  // operation.
  if (lhs && rhs) {
    return factoryFn(opFn(lhs, rhs));
  }

  // Otherwise, do nothing.
  return nullptr;
}

/// Attempts to fold a binary operation over felts.
///
/// If the operation does not have access to the prime field, is not folded.
template <typename Op, typename Fn>
OpFoldResult tryFoldBinaryFeltOp(
    Op &op, typename Op::FoldAdaptor adaptor, Fn opFn,
    llvm::function_ref<bool(const APInt &, Side)> isIdentity,
    llvm::function_ref<bool(const APInt &, Side)> isZero
) {
  auto prime = getFieldPrime(op);
  if (!prime) {
    return nullptr;
  }

  return foldBinaryOp<FeltAttr>(op, adaptor, [&prime, opFn](FeltAttr lhs, FeltAttr rhs) {
    return FeltAttr::get(
        lhs.getContext(),
        opFn(prime->reduce(lhs).getValue(), prime->reduce(rhs).getValue(), prime->getValue())
    );
  }, [&prime, isIdentity](FeltAttr value, auto side) {
    return isIdentity(prime->reduce(value).getValue(), side);
  }, [&prime, isZero](FeltAttr value, auto side) {
    return isZero(prime->reduce(value).getValue(), side);
  }, [&prime](auto value) { return prime->reduce(value); });
}

/// Attempts to fold a comparison operation over felts.
template <typename Op, typename Fn>
OpFoldResult foldCmpOp(Op &op, typename Op::FoldAdaptor adaptor, Fn opFn) {
  auto prime = getFieldPrime(op);
  if (!prime) {
    return nullptr;
  }
  auto lhs = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getLhs());
  auto rhs = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getRhs());
  // Shortcircuit if either operand is not constant.
  if (!rhs || !lhs) {
    return nullptr;
  }
  lhs = prime->reduce(lhs);
  rhs = prime->reduce(rhs);

  return pcl::BoolAttr::get(op->getContext(), opFn(lhs.getValue(), rhs.getValue()));
}

/// Helper for doing unsigned addition on field elements represented by `APInt`s.
///
/// Adjusts the bit width to the correct size before adding. The caller is
/// responsible of wrapping the value back into the field if it's intended to
/// continue representing a field element.
static APInt safeAdd(const APInt &lhs, const APInt &rhs) {
  auto w = std::max({lhs.getBitWidth(), rhs.getBitWidth()}) + 1;
  auto lhsExt = lhs.zext(w);
  auto rhsExt = rhs.zext(w);
  return lhsExt + rhsExt;
}

/// Helper for doing unsigned subtraction on field elements represented by `APInt`s.
///
/// Adjusts the bit width to the correct size before subtracting. The caller is
/// responsible of wrapping the value back into the field if it's intended to
/// continue representing a field element.
static APInt safeSub(const APInt &lhs, const APInt &rhs) {
  auto w = std::max({lhs.getBitWidth(), rhs.getBitWidth()}) + 1;
  auto lhsExt = lhs.zext(w);
  auto rhsExt = rhs.zext(w);
  return lhsExt - rhsExt;
}

/// Helper for doing unsigned multiplication on field elements represented by `APInt`s.
///
/// Adjusts the bit width to the correct size before multiplying. The caller is
/// responsible of wrapping the value back into the field if it's intended to
/// continue representing a field element.
static APInt safeMul(const APInt &lhs, const APInt &rhs) {
  /// Add an extra +1 just to be safe.
  auto w = lhs.getBitWidth() + rhs.getBitWidth() + 1;
  auto lhsExt = lhs.zext(w);
  auto rhsExt = rhs.zext(w);
  return lhsExt * rhsExt;
}

/// Pattern for folding "double negations". It is generalized to any pattern
/// in the form `(op (op X))`.
///
/// The pattern is only applied if both ops have 1 operand.
template <typename Op> struct FoldDoubleNeg : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) {
      return failure();
    }
    auto defOp = mlir::dyn_cast_if_present<Op>(op->getOperand(0).getDefiningOp());
    if (!defOp || defOp->getNumOperands() != 1) {
      return failure();
    }
    rewriter.replaceOp(op, ValueRange(defOp->getOperand(0)));

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Ops over fields (In alphabetical order)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return tryFoldBinaryFeltOp(*this, adaptor, [](const auto &lhs, const auto &rhs, const auto &) {
    return safeAdd(lhs, rhs);
  }, [](const auto &value, auto) { return value.isZero(); }, [](const auto &, auto) {
    return false;
  });
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) {
  auto prime = getFieldPrime(*this);
  if (!prime) {
    return adaptor.getValue();
  }
  return prime->reduce(adaptor.getValue());
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

namespace {

/// Rewrites multiplication operations of the form `X * (p - 1)`
/// to `-X`
struct RewriteXTimesMinus1ToNegX : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const override {
    auto prime = getFieldPrime(op);
    if (!prime) {
      return failure();
    }
    auto value = matchOperands(op, *prime);
    if (failed(value)) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<NegOp>(op, *value);
    return success();
  }

private:
  FailureOr<Value> matchOperands(MulOp op, PrimeAttr prime) const {
    auto lhs = matchOperandsImpl(op.getLhs(), op.getRhs(), prime);
    if (succeeded(lhs)) {
      return lhs;
    }
    return matchOperandsImpl(op.getRhs(), op.getLhs(), prime);
  }

  FailureOr<Value> matchOperandsImpl(Value lhs, Value rhs, PrimeAttr prime) const {
    auto feltAttr = getAttr(lhs);
    if (!feltAttr) {
      return failure();
    }
    // If p - v != 1, ignore this case.
    if (!safeSub(prime.getValue(), prime.reduce(feltAttr).getValue()).isOne()) {
      return failure();
    }

    return rhs;
  }

  FeltAttr getAttr(Value v) const {
    Attribute attr;
    if (!matchPattern(v, m_Constant(&attr))) {
      return FeltAttr();
    }
    return mlir::dyn_cast_if_present<FeltAttr>(attr);
  }
};
} // namespace

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  return tryFoldBinaryFeltOp(*this, adaptor, [](const auto &lhs, const auto &rhs, const auto &) {
    return safeMul(lhs, rhs);
  }, [](auto &value, auto) { return value.isOne(); }, [](auto &value, auto) {
    return value.isZero();
  });
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RewriteXTimesMinus1ToNegX>(context);
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  auto prime = getFieldPrime(*this);
  if (!prime) {
    return nullptr;
  }
  auto attr = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getValue());
  if (!attr) {
    return nullptr;
  }
  return prime->reduce(-attr.getValue());
}

void NegOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldDoubleNeg<NegOp>>(context);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

namespace {

/// Converts subtraction operations where LHS is a constant 0 into the negation of RHS.
struct ZeroMinusXToNegX : public OpRewritePattern<SubOp> {
  using OpRewritePattern<SubOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SubOp op, PatternRewriter &rewriter) const override {
    auto lhsAttr = getLhsAttr(op);
    if (!lhsAttr || !lhsAttr.getValue().isZero()) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<NegOp>(op, op.getRhs());
    return success();
  }

private:
  FeltAttr getLhsAttr(SubOp op) const {
    Attribute attr;
    if (!matchPattern(op.getLhs(), m_Constant(&attr))) {
      return FeltAttr();
    }
    return mlir::dyn_cast_if_present<FeltAttr>(attr);
  }
};

} // namespace

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ZeroMinusXToNegX>(context);
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return tryFoldBinaryFeltOp(
      *this, adaptor, [](const auto &lhs, const auto &rhs, const auto &prime) {
    // (lhs - rhs) mod p == (lhs + (p - rhs)) mod p iff 0 <= lhs < p and 0 <= rhs < p.
    // The `tryFoldBinaryFeltOp` helper ensures `lhs` and `rhs` are inside the field, so the
    // assumption above is safe.
    return safeAdd(lhs, safeSub(prime, rhs));
  }, [](auto &value, auto side) {
    // lhs - 0 = lhs
    return side == Side::Rhs && value.isZero();
  }, [](auto &, auto) { return false; }
  );
}

//===----------------------------------------------------------------------===//
// Comparison ops
//===----------------------------------------------------------------------===//

namespace {

/// Helper for the `FoldEqBoolean` pattern.
struct FoldedEq {
  /// The value of the other operand.
  Value value;
  /// Whether the compared value was 0 or 1.
  bool constValue;
};

/// Folds the following cases to simplify the IR.
///
/// - `(= 1 (asfelt X))` => `X`
/// - `(= (asfelt X) 1)` => `X`
/// - `(= 0 (asfelt X))` => `(not X)`
/// - `(= (asfelt X) 0)` => `(not X)`
/// Assuming `X` is of type `!pcl.bool`
struct FoldEqBoolean : public OpRewritePattern<CmpEqOp> {
  using OpRewritePattern<CmpEqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpEqOp op, PatternRewriter &rewriter) const override {
    auto folded = matchAssertion(op);
    if (failed(folded)) {
      return failure();
    }

    Value x = folded->value;
    if (!folded->constValue) {
      // The constant is false.
      x = rewriter.createOrFold<NotOp>(op.getLoc(), x);
    }
    rewriter.replaceOp(op, x);
    return success();
  }

private:
  /// Matches the assertion to a pattern.
  FailureOr<FoldedEq> matchAssertion(CmpEqOp op) const {
    auto lhsMatch = matchAssertionImpl(op.getLhs(), op.getRhs());
    if (succeeded(lhsMatch)) {
      return lhsMatch;
    }

    return matchAssertionImpl(op.getRhs(), op.getLhs());
  }

  /// Simpler pattern that assumes only LHS can be the constant.
  FailureOr<FoldedEq> matchAssertionImpl(Value lhs, Value rhs) const {
    Attribute attr;
    auto rhsAsBool = mlir::dyn_cast_if_present<AsFeltOp>(rhs.getDefiningOp());

    if (!matchPattern(lhs, m_Constant(&attr)) || !rhsAsBool) {
      return failure();
    }

    if (auto boolAttr = mlir::dyn_cast_if_present<pcl::BoolAttr>(attr)) {
      return FoldedEq {.value = rhsAsBool.getValue(), .constValue = boolAttr.getValue()};
    }

    if (auto feltAttr = mlir::dyn_cast_if_present<FeltAttr>(attr)) {
      if (feltAttr.getValue().isZero()) {
        return FoldedEq {.value = rhsAsBool.getValue(), .constValue = false};
      }
      if (feltAttr.getValue().isOne()) {
        return FoldedEq {.value = rhsAsBool.getValue(), .constValue = true};
      }
    }
    return failure();
  }
};

/// Folds `(= (- X) (- Y))` into `(= X Y)`
struct FoldEqOfNegations : public OpRewritePattern<CmpEqOp> {
  using OpRewritePattern<CmpEqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpEqOp op, PatternRewriter &rewriter) const override {
    auto lhsNeg = mlir::dyn_cast_if_present<NegOp>(op.getLhs().getDefiningOp());
    auto rhsNeg = mlir::dyn_cast_if_present<NegOp>(op.getRhs().getDefiningOp());

    if (!lhsNeg || !rhsNeg) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<CmpEqOp>(op, lhsNeg.getValue(), rhsNeg.getValue());
    return success();
  }
};

/// Folds `(= 0 (+ X Y))` into `(= X (- Y))`.
///
/// When picking which side to move to the other side the pattern gives priority to
/// negation ops.
///
/// This pattern helps simplifying constraints generated for some constraint systems
/// that are asserted to be equal to 0.
///
/// For example, the constraint `X * Y = Z` is represented as `X * Y - Z = 0`. This pattern
/// rewrites it to an equality with terms in both sides.
struct FoldEqOfZeroAndSum : public OpRewritePattern<CmpEqOp> {
  using OpRewritePattern<CmpEqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpEqOp op, PatternRewriter &rewriter) const override {
    auto addOp = matchAssertion(op);
    if (!addOp) {
      return failure();
    }
    auto [X, Y] = pickOperands(addOp, rewriter);
    rewriter.replaceOpWithNewOp<CmpEqOp>(op, X, Y);
    return success();
  }

private:
  /// Picks the operands of the add op `X` and `Y` for creating
  /// the new eq operation as `(= X Y)`
  std::pair<Value, Value> pickOperands(AddOp op, PatternRewriter &rewriter) const {
    auto X = op.getLhs();
    auto Y = op.getRhs();
    auto Xop = mlir::dyn_cast_if_present<NegOp>(X.getDefiningOp());
    auto Yop = mlir::dyn_cast_if_present<NegOp>(Y.getDefiningOp());
    if (Xop) {
      return {Xop.getValue(), Y};
    }
    if (Yop) {
      return {X, Yop.getValue()};
    }

    auto negOp = rewriter.create<NegOp>(op.getLoc(), Y);
    return {X, negOp};
  }

  /// Matches the assertion to a pattern.
  AddOp matchAssertion(CmpEqOp op) const {
    auto lhsMatch = matchAssertionImpl(op.getLhs(), op.getRhs());
    if (!lhsMatch) {
      return lhsMatch;
    }

    return matchAssertionImpl(op.getRhs(), op.getLhs());
  }

  /// Simpler pattern that assumes only LHS can be the constant.
  AddOp matchAssertionImpl(Value lhs, Value rhs) const {
    Attribute attr;

    if (!matchPattern(lhs, m_Constant(&attr))) {
      return AddOp();
    }
    auto feltAttr = mlir::dyn_cast_if_present<FeltAttr>(attr);
    if (!feltAttr || !feltAttr.getValue().isZero()) {
      return AddOp();
    }

    return mlir::dyn_cast_if_present<AddOp>(rhs.getDefiningOp());
  }
};

/// Folds `(= 0 (- X Y))` into `(= X Y)`.
///
/// The goal of this pattern is similar to `FoldEqOfZeroAndSum` but applied to subtractions.
struct FoldEqOfZeroAndSub : public OpRewritePattern<CmpEqOp> {
  using OpRewritePattern<CmpEqOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CmpEqOp op, PatternRewriter &rewriter) const override {
    auto subOp = matchAssertion(op);
    if (!subOp) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<CmpEqOp>(op, subOp.getLhs(), subOp.getRhs());
    return success();
  }

private:
  /// Matches the assertion to a pattern.
  SubOp matchAssertion(CmpEqOp op) const {
    auto lhsMatch = matchAssertionImpl(op.getLhs(), op.getRhs());
    if (!lhsMatch) {
      return lhsMatch;
    }

    return matchAssertionImpl(op.getRhs(), op.getLhs());
  }

  /// Simpler pattern that assumes only LHS can be the constant.
  SubOp matchAssertionImpl(Value lhs, Value rhs) const {
    Attribute attr;

    if (!matchPattern(lhs, m_Constant(&attr))) {
      return SubOp();
    }
    auto feltAttr = mlir::dyn_cast_if_present<FeltAttr>(attr);
    if (!feltAttr || !feltAttr.getValue().isZero()) {
      return SubOp();
    }

    return mlir::dyn_cast_if_present<SubOp>(rhs.getDefiningOp());
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// CmpEqOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpEqOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](const auto &lhs, const auto &rhs) { return lhs == rhs; });
}

void CmpEqOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldEqBoolean, FoldEqOfNegations, FoldEqOfZeroAndSum, FoldEqOfZeroAndSub>(context);
}

//===----------------------------------------------------------------------===//
// CmpLtOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpLtOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](const auto &lhs, const auto &rhs) { return lhs.ult(rhs); });
}

//===----------------------------------------------------------------------===//
// CmpLeOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpLeOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](const auto &lhs, const auto &rhs) { return lhs.ule(rhs); });
}

//===----------------------------------------------------------------------===//
// CmpGtOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpGtOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](const auto &lhs, const auto &rhs) { return lhs.ugt(rhs); });
}

//===----------------------------------------------------------------------===//
// CmpGeOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpGeOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](const auto &lhs, const auto &rhs) { return lhs.uge(rhs); });
}

//===----------------------------------------------------------------------===//
// Ops over formulas (In alphabetical order)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOp<BoolAttr>(*this, adaptor, [](auto lhs, auto rhs) {
    return BoolAttr::get(lhs.getContext(), lhs.getValue() && rhs.getValue());
  }, [](auto value, auto) { return value.getValue(); }, [](auto value, auto) {
    return !value.getValue();
  });
}

//===----------------------------------------------------------------------===//
// AsFeltOp
//===----------------------------------------------------------------------===//

/// If the boolean is constant, fold the op into a constant 1 or 0.
OpFoldResult AsFeltOp::fold(FoldAdaptor adaptor) {
  auto attr = mlir::dyn_cast_if_present<BoolAttr>(adaptor.getValue());
  if (!attr) {
    return nullptr;
  }
  auto prime = getFieldPrime(*this);
  // If the prime is not available use BW=2. Once the prime is available other folding operations
  // will take care of adjusting the width.
  return FeltAttr::get(
      getContext(), APInt(prime ? prime->getBitWidth() : 2, attr.getValue() ? 1 : 0)
  );
}

//===----------------------------------------------------------------------===//
// DetOp
//===----------------------------------------------------------------------===//

/// Fold the det operation if the operand is constant, since it's going to be
/// deterministic by definition.
OpFoldResult DetOp::fold(FoldAdaptor adaptor) {
  auto attr = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getValue());
  if (!attr) {
    return nullptr;
  }
  return BoolAttr::get(getContext(), true);
}

//===----------------------------------------------------------------------===//
// FalseOp
//===----------------------------------------------------------------------===//

OpFoldResult FalseOp::fold(FoldAdaptor) { return BoolAttr::get(getContext(), false); }

//===----------------------------------------------------------------------===//
// IffOp
//===----------------------------------------------------------------------===//

OpFoldResult IffOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOp<BoolAttr>(*this, adaptor, [](auto lhs, auto rhs) {
    return BoolAttr::get(lhs.getContext(), lhs.getValue() == rhs.getValue());
  }, [](auto, auto) { return false; }, [](auto, auto) { return false; });
}

//===----------------------------------------------------------------------===//
// ImpliesOp
//===----------------------------------------------------------------------===//

OpFoldResult ImpliesOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOp<BoolAttr>(*this, adaptor, [](auto lhs, auto rhs) {
    return BoolAttr::get(lhs.getContext(), !lhs.getValue() || rhs.getValue());
  }, [](auto, auto) { return false; }, [](auto value, auto side) {
    // p -> T = T
    return value.getValue() && side == Side::Rhs;
  });
}

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

OpFoldResult NotOp::fold(FoldAdaptor adaptor) {
  auto attr = mlir::dyn_cast_if_present<BoolAttr>(adaptor.getValue());
  if (!attr) {
    return nullptr;
  }

  return BoolAttr::get(getContext(), !attr.getValue());
}

void NotOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldDoubleNeg<NotOp>>(context);
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOp<BoolAttr>(*this, adaptor, [](auto lhs, auto rhs) {
    return BoolAttr::get(lhs.getContext(), lhs.getValue() || rhs.getValue());
  }, [](auto value, auto) { return !value.getValue(); }, [](auto value, auto) {
    return value.getValue();
  });
}

//===----------------------------------------------------------------------===//
// TrueOp
//===----------------------------------------------------------------------===//

OpFoldResult TrueOp::fold(FoldAdaptor) { return BoolAttr::get(getContext(), true); }

//===----------------------------------------------------------------------===//
// Assertions & post-conditions operations
//===----------------------------------------------------------------------===//

namespace {

/// Removes any assert-like operation whose condition is a constant true.
template <typename Op> struct RemoveTauto : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const override {
    auto condAttr = getCondAttr(op);
    if (!condAttr || !condAttr.getValue()) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  pcl::BoolAttr getCondAttr(Op op) const {
    Attribute attr;
    if (!matchPattern(op.getCond(), m_Constant(&attr))) {
      return pcl::BoolAttr();
    }
    return mlir::dyn_cast_if_present<pcl::BoolAttr>(attr);
  }
};

template <typename Op>
void addAssertLikeCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RemoveTauto<Op>>(context);
}
} // namespace

//===----------------------------------------------------------------------===//
// AssertOp
//===----------------------------------------------------------------------===//

void AssertOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  addAssertLikeCanonicalizationPatterns<AssertOp>(patterns, context);
}

//===----------------------------------------------------------------------===//
// PostOp
//===----------------------------------------------------------------------===//

void PostOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  addAssertLikeCanonicalizationPatterns<PostOp>(patterns, context);
}
