//===-- Compare.h -----------------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/Operation.h>

#include <concepts>
#include <utility>

namespace llzk {

template <typename Op>
concept OpComparable = requires(Op op) {
  { op.getOperation() } -> std::convertible_to<mlir::Operation *>;
};

template <typename Op>
concept NamedOpComparable = OpComparable<Op> && requires(Op op) {
  { op.getName() } -> std::convertible_to<mlir::StringRef>;
};

struct FileLineColLocComparator {
  bool operator()(const mlir::FileLineColLoc &LHS, const mlir::FileLineColLoc &RHS) const {
    auto filenameCmp = LHS.getFilename().compare(RHS.getFilename());
    return filenameCmp < 0 || (filenameCmp == 0 && LHS.getLine() < RHS.getLine()) ||
           (filenameCmp == 0 && LHS.getLine() == RHS.getLine() &&
            LHS.getColumn() < RHS.getColumn());
  }
};

struct LocationComparator {
  bool operator()(const mlir::Location &LHS, const mlir::Location &RHS) const {
    auto lhsFileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(LHS);
    auto rhsFileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(RHS);
    if (lhsFileLoc && rhsFileLoc) {
      return FileLineColLocComparator {}(lhsFileLoc, rhsFileLoc);
    }
    return mlir::hash_value(LHS) < mlir::hash_value(RHS);
  }
};

/// Compare the source locations of two operations.
///
/// Returns failure unless both operations have `FileLineColLoc` locations. A successful `false`
/// result means only that `l` is not before `r`; the locations may be equal. This comparison is
/// therefore not a total order.
template <OpComparable Op> mlir::FailureOr<bool> isLocationLess(const Op &l, const Op &r) {
  mlir::Location lhsLoc = l->getLoc(), rhsLoc = r->getLoc();
  // We cannot make judgments on unknown locations.
  if (llvm::isa<mlir::UnknownLoc>(lhsLoc) || llvm::isa<mlir::UnknownLoc>(rhsLoc)) {
    return mlir::failure();
  }
  // If we have full locations for both, then we can sort by file name, then line, then column.
  auto lhsFileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(lhsLoc);
  auto rhsFileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(rhsLoc);
  if (lhsFileLoc && rhsFileLoc) {
    return FileLineColLocComparator {}(lhsFileLoc, rhsFileLoc);
  }
  return mlir::failure();
}

template <OpComparable Op> struct OpLocationLess {
  bool operator()(const Op &l, const Op &r) const { return isLocationLess(l, r).value_or(false); }
};

/// Order named operations by source location, using the symbol name to break ties or when source
/// locations cannot be compared.
template <NamedOpComparable Op> struct NamedOpLocationLess {
  bool operator()(const Op &l, const Op &r) const {
    auto res = isLocationLess(l, r);
    auto reverseRes = isLocationLess(r, l);
    // A successful false result may mean either equal locations or that the operands are reversed.
    // Check both directions before using the symbol name as the deterministic tie-breaker.
    if (mlir::succeeded(res) && res.value()) {
      return true;
    }
    if (mlir::succeeded(reverseRes) && reverseRes.value()) {
      return false;
    }

    Op &lhs = const_cast<Op &>(l);
    Op &rhs = const_cast<Op &>(r);
    return lhs.getName().compare(rhs.getName()) < 0;
  }
};

template <typename T, typename U> constexpr T checkedCast(U u) noexcept {
  assert(std::in_range<T>(u) && "lossy conversion");
  return static_cast<T>(u);
}

} // namespace llzk
