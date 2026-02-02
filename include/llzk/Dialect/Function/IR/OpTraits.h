//===-- OpTraits.h ----------------------------------------------*- c++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#pragma once

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LogicalResult.h>

namespace llzk::function {

mlir::LogicalResult verifyConstraintGenTraitImpl(mlir::Operation *op);
mlir::LogicalResult verifyWitnessGenTraitImpl(mlir::Operation *op);
mlir::LogicalResult verifyNotFieldNativeTraitImpl(mlir::Operation *op);

/// Marker for ops that are specific to constraint generation.
/// Verifies that the surrounding function is marked with the `AllowConstraintAttr`.
template <typename TypeClass>
class ConstraintGen : public mlir::OpTrait::TraitBase<TypeClass, ConstraintGen> {
public:
  inline static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return verifyConstraintGenTraitImpl(op);
  }
};

/// Marker for ops that are specific to witness generation.
/// Verifies that the surrounding function is marked with the `AllowWitnessAttr`.
template <typename TypeClass>
class WitnessGen : public mlir::OpTrait::TraitBase<TypeClass, WitnessGen> {
public:
  inline static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return verifyWitnessGenTraitImpl(op);
  }
};

/// Marker for ops over `llzk.felt` type operands that are not native to finite field arithmetic.
/// Verifies that the surrounding function is marked with the `AllowNonNativeFieldOpsAttr`.
template <typename TypeClass>
class NotFieldNative : public mlir::OpTrait::TraitBase<TypeClass, NotFieldNative> {
public:
  inline static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return verifyNotFieldNativeTraitImpl(op);
  }
};

} // namespace llzk::function
