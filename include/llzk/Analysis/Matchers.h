//===-- Matchers.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Struct/IR/Ops.h"

#include <mlir/IR/Matchers.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace llzk {

/// @brief This matcher will either match on `lhs op rhs` or `rhs op lhs`. If
/// `LhsMatcher` == `RhsMatcher`, using this matcher is unnecessary.
/// @tparam ...OpTypes The types of ops that can be matched.
template <typename LhsMatcher, typename RhsMatcher, typename... OpTypes> struct CommutativeMatcher {

  CommutativeMatcher(LhsMatcher lhs, RhsMatcher rhs) : lhsMatcher(lhs), rhsMatcher(rhs) {}

  bool match(mlir::Operation *op) {
    using namespace mlir::detail;
    if (!isa<OpTypes...>(op) || op->getNumOperands() != 2) {
      return false;
    }
    bool res = matchOperandOrValueAtIndex(op, 0, lhsMatcher) &&
               matchOperandOrValueAtIndex(op, 1, rhsMatcher);
    if (!res) {
      res = matchOperandOrValueAtIndex(op, 1, lhsMatcher) &&
            matchOperandOrValueAtIndex(op, 0, rhsMatcher);
    }
    return res;
  }

  LhsMatcher lhsMatcher;
  RhsMatcher rhsMatcher;
};

template <typename OpType, typename LhsMatcher, typename RhsMatcher>
auto m_CommutativeOp(LhsMatcher lhs, RhsMatcher rhs) {
  return CommutativeMatcher<LhsMatcher, RhsMatcher, OpType>(lhs, rhs);
}

/// @brief Matches and optionally captures a SourceRef base value, which is either
/// a member read or a block argument (i.e., an input to a @constrain or @compute function).
struct RefValueCapture {
  mlir::Value *what;
  RefValueCapture(mlir::Value *capture) : what(capture) {}

  bool match(mlir::Value v) {
    if (isa<mlir::BlockArgument>(v) ||
        isa_and_present<component::MemberReadOp>(v.getDefiningOp())) {
      if (what) {
        *what = v;
      }
      return true;
    }
    return false;
  }
};

auto m_RefValue() { return RefValueCapture(nullptr); }

auto m_RefValue(mlir::Value *capture) { return RefValueCapture(capture); }

/// @brief Matches and optionally captures a felt constant.
struct ConstantCapture {
  felt::FeltConstantOp *what;
  ConstantCapture(felt::FeltConstantOp *capture) : what(capture) {}

  bool match(mlir::Value v) {
    if (auto match = dyn_cast_if_present<felt::FeltConstantOp>(v.getDefiningOp())) {
      if (what) {
        *what = match;
      }
      return true;
    }
    return false;
  }
};

auto m_Constant() { return ConstantCapture(nullptr); }

auto m_Constant(felt::FeltConstantOp *capture) { return ConstantCapture(capture); }

} // namespace llzk
