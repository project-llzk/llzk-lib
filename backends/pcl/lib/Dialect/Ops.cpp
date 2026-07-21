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

/// Folds a binary operation.
///
/// The helper is generic over the operation based on a set of callbacks:
/// - A callback that defines the actual operation
/// - A callback that queries if the given value is the identy under that operation (i.e. 1 for
/// multiplication or 0 for addition).
/// - A callback that queries if the given value "cancels out" the operation (i.e. 0 for
/// multiplication or false for conjunction).
/// - An optional callback that creates the fold result.
template <typename T, typename Op>
OpFoldResult foldBinaryOp(
    Op &op, typename Op::FoldAdaptor adaptor, llvm::function_ref<T(T, T)> opFn,
    llvm::function_ref<bool(T)> isIdentity, llvm::function_ref<bool(T)> isZero,
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
  if (lhs && isZero(lhs)) {
    return factoryFn(lhs);
  }
  if (rhs && isZero(rhs)) {
    return factoryFn(rhs);
  }
  // If either side is the identity, return the other side.
  // If the other side is a constant, return the attribute representing it.
  // Otherwise, return the value of the operand.
  if (lhs && isIdentity(lhs)) {
    return attrOrValue(rhs, op.getRhs());
  }
  if (rhs && isIdentity(rhs)) {
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
template <typename Op>
OpFoldResult tryFoldBinaryFeltOp(
    Op &op, typename Op::FoldAdaptor adaptor, llvm::function_ref<FeltAttr(FeltAttr, FeltAttr)> opFn,
    llvm::function_ref<bool(FeltAttr)> isIdentity, llvm::function_ref<bool(FeltAttr)> isZero
) {
  auto prime = getFieldPrime(op);
  if (!prime) {
    return nullptr;
  }

  return foldBinaryOp<FeltAttr>(op, adaptor, opFn, isIdentity, isZero, [&prime](auto value) {
    return prime->reduce(value);
  });
}

/// Attempts to fold a comparison operation over felts.
template <typename Op>
OpFoldResult foldCmpOp(
    Op &op, typename Op::FoldAdaptor adaptor, llvm::function_ref<bool(FeltAttr, FeltAttr)> opFn
) {
  auto lhs = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getLhs());
  auto rhs = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getRhs());
  // Shortcircuit if either operand is not constant.
  if (!rhs || !lhs) {
    return nullptr;
  }

  return pcl::BoolAttr::get(op->getContext(), opFn(lhs, rhs));
}

} // namespace

//===----------------------------------------------------------------------===//
// Ops over fields (In alphabetical order)
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return tryFoldBinaryFeltOp(*this, adaptor, [](auto lhs, auto rhs) {
    return FeltAttr::get(lhs.getContext(), lhs.getValue() + rhs.getValue());
  }, [](auto value) { return value.getValue().isZero(); }, [](auto) { return false; });
}

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstOp::fold(FoldAdaptor adaptor) { return adaptor.getValue(); }

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldXTimesMinus1 : public OpRewritePattern<MulOp> {
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
    auto diff = prime.getValue() - feltAttr.getValue();
    llvm::dbgs() << "Prime: " << prime.getValue() << "\nValue: " << feltAttr.getValue()
                 << "\nDiff: " << diff << "\n";
    if (!diff.isOne()) {
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
  return tryFoldBinaryFeltOp(*this, adaptor, [](auto lhs, auto rhs) {
    return FeltAttr::get(lhs.getContext(), lhs.getValue() * rhs.getValue());
  }, [](auto value) { return value.getValue().isOne(); }, [](auto value) {
    return value.getValue().isZero();
  });
}

void MulOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldXTimesMinus1>(context);
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  auto prime = getFieldPrime(*this);
  if (!prime) {
    return nullptr;
  }
  auto attr = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getVal());
  if (!attr) {
    return nullptr;
  }
  auto negated = attr.getValue() * -1;
  return prime->reduce(attr.getValue() * -1);
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
  return tryFoldBinaryFeltOp(*this, adaptor, [](auto lhs, auto rhs) {
    return FeltAttr::get(lhs.getContext(), lhs.getValue() - rhs.getValue());
  }, [](auto) { return false; }, [](auto) { return false; });
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
/// - `(= 1 X)` => `X`
/// - `(= X 1)` => `X`
/// - `(= 0 X)` => `(not X)`
/// - `(= X 0)` => `(not X)`
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
      return FoldedEq {.value = rhsAsBool.getVal(), .constValue = boolAttr.getValue()};
    }

    if (auto feltAttr = mlir::dyn_cast_if_present<FeltAttr>(attr)) {
      if (feltAttr.getValue().isZero()) {
        return FoldedEq {.value = rhsAsBool.getVal(), .constValue = false};
      }
      if (feltAttr.getValue().isOne()) {
        return FoldedEq {.value = rhsAsBool.getVal(), .constValue = true};
      }
    }
    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// CmpEqOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpEqOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](auto lhs, auto rhs) {
    return lhs.getValue() == rhs.getValue();
  });
}

void CmpEqOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldEqBoolean>(context);
}

//===----------------------------------------------------------------------===//
// CmpLtOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpLtOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](auto lhs, auto rhs) {
    return lhs.getValue().ult(rhs.getValue());
  });
}

//===----------------------------------------------------------------------===//
// CmpLeOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpLeOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](auto lhs, auto rhs) {
    return lhs.getValue().ule(rhs.getValue());
  });
}

//===----------------------------------------------------------------------===//
// CmpGtOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpGtOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](auto lhs, auto rhs) {
    return lhs.getValue().ugt(rhs.getValue());
  });
}

//===----------------------------------------------------------------------===//
// CmpGeOp
//===----------------------------------------------------------------------===//

OpFoldResult CmpGeOp::fold(FoldAdaptor adaptor) {
  return foldCmpOp(*this, adaptor, [](auto lhs, auto rhs) {
    return lhs.getValue().uge(rhs.getValue());
  });
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
  }, [](auto value) { return value.getValue(); }, [](auto value) { return !value.getValue(); });
}

//===----------------------------------------------------------------------===//
// AsFeltOp
//===----------------------------------------------------------------------===//

/// If the boolean is constant, fold the op into a constant 1 or 0.
OpFoldResult AsFeltOp::fold(FoldAdaptor adaptor) {
  auto prime = getFieldPrime(*this);
  if (!prime) {
    return nullptr;
  }
  auto attr = mlir::dyn_cast_if_present<BoolAttr>(adaptor.getVal());
  if (!attr) {
    return nullptr;
  }

  return FeltAttr::get(getContext(), APInt(prime->getBitWidth(), attr.getValue() ? 1 : 0));
}

//===----------------------------------------------------------------------===//
// DetOp
//===----------------------------------------------------------------------===//

/// Fold the det operation if the operand is constant, since it's going to be
/// deterministic by definition.
OpFoldResult DetOp::fold(FoldAdaptor adaptor) {
  auto attr = mlir::dyn_cast_if_present<FeltAttr>(adaptor.getExpr());
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
  }, [](auto) { return false; }, [](auto) { return false; });
}

//===----------------------------------------------------------------------===//
// ImpliesOp
//===----------------------------------------------------------------------===//

OpFoldResult ImpliesOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOp<BoolAttr>(*this, adaptor, [](auto lhs, auto rhs) {
    return BoolAttr::get(lhs.getContext(), !lhs.getValue() || rhs.getValue());
  }, [](auto) { return false; }, [](auto) { return false; });
}

//===----------------------------------------------------------------------===//
// NotOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldDoubleNeg : public OpRewritePattern<NotOp> {
  using OpRewritePattern<NotOp>::OpRewritePattern;

  LogicalResult match(NotOp op) const override {
    return success(mlir::isa_and_present<NotOp>(op.getCond().getDefiningOp()));
  }

  void rewrite(NotOp op, PatternRewriter &rewriter) const override {
    auto cond = mlir::cast<NotOp>(op.getCond().getDefiningOp());
    rewriter.replaceOp(op, cond.getCond());
  }
};
} // namespace

OpFoldResult NotOp::fold(FoldAdaptor adaptor) {
  auto attr = mlir::dyn_cast_if_present<BoolAttr>(adaptor.getCond());
  if (!attr) {
    return nullptr;
  }

  return BoolAttr::get(getContext(), !attr.getValue());
}

void NotOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FoldDoubleNeg>(context);
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  return foldBinaryOp<BoolAttr>(*this, adaptor, [](auto lhs, auto rhs) {
    return BoolAttr::get(lhs.getContext(), lhs.getValue() || rhs.getValue());
  }, [](auto value) { return !value.getValue(); }, [](auto value) { return value.getValue(); });
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

  LogicalResult match(Op op) const override {
    auto condAttr = getCondAttr(op);
    return failure(!condAttr || !condAttr.getValue());
  }

  void rewrite(Op op, PatternRewriter &rewriter) const override { rewriter.eraseOp(op); }

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
