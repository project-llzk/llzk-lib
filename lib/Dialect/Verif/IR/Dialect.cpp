//===-- Dialect.cpp - Verif dialect implementation --------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Verif/IR/Dialect.h"

#include "llzk/Dialect/LLZK/IR/Versioning.h"
#include "llzk/Dialect/Verif/IR/Ops.h"

#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/TypeID.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/TypeSwitch.h>

// TableGen'd implementation files
#include "llzk/Dialect/Verif/IR/Dialect.cpp.inc"

using namespace mlir;

//===------------------------------------------------------------------===//
// InvariantTarget implementations for upstream dialects
//===------------------------------------------------------------------===//

namespace {

/// Shared implementation of the `getLabel` method.
static FailureOr<StringRef> getLabelImpl(Operation *op) {
  auto attr = dyn_cast_or_null<StringAttr>(op->getDiscardableAttr("loop_label"));
  if (!attr) {
    return failure();
  }
  return attr.getValue();
}

struct ScfWhileExternalModel : public llzk::verif::InvariantTargetOpInterface::ExternalModel<
                                   ScfWhileExternalModel, scf::WhileOp> {
  FailureOr<StringRef> getLabel(Operation *op) const { return getLabelImpl(op); }

  SmallVector<Type> getArgumentTypes(Operation *op) const {
    auto whileOp = cast<scf::WhileOp>(op);
    // In the case of `scf.while` we return the 'before' arguments.
    // Depending on how the loop is constructed it may not be the most ergonomic
    // when it comes to binding the loop arguments in an invariant. The limitation of using these
    // arguments is that invariants will have trouble expressing properties of a loop that rely
    // on intermediate values passed via the 'after' arguments. The downside of using both
    // 'before' and 'after' arguments is that any 'before' argument that is passed to the
    // 'after' arguments will require a duplicate binding in the invariant, which is probably
    // not very user-friendly and may lead to confusion.
    return llvm::map_to_vector(whileOp.getBeforeArguments(), [](auto arg) {
      return arg.getType();
    });
  }
};

struct ScfForExternalModel : public llzk::verif::InvariantTargetOpInterface::ExternalModel<
                                 ScfForExternalModel, scf::ForOp> {
  FailureOr<StringRef> getLabel(Operation *op) const { return getLabelImpl(op); }

  SmallVector<Type> getArgumentTypes(Operation *op) const {
    auto forOp = cast<scf::ForOp>(op);
    // In the case of `scf.for` we return the control values in a fixed order defined by the
    // spec language semantics followed by any loop carried values. The order for the control values
    // is: lower bound, induction variable, upper bound, step.
    SmallVector<Type, 4> types;
    types.reserve(forOp.getNumRegionIterArgs() + 4);
    types.append(
        {forOp.getLowerBound().getType(), forOp.getInductionVar().getType(),
         forOp.getUpperBound().getType(), forOp.getStep().getType()}
    );
    types.append(llvm::map_to_vector(forOp.getInitArgs(), [](auto arg) { return arg.getType(); }));
    return types;
  }
};

/// Dialect extension that attaches the interfaces to upstream ops that promised them.
struct InterfacesExtension
    : public DialectExtension<InterfacesExtension, llzk::verif::VerifDialect, scf::SCFDialect> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InterfacesExtension)

  void apply(MLIRContext *context, llzk::verif::VerifDialect *, scf::SCFDialect *) const final {
    llzk::verif::attachInterfaces(*context);
  }
};

} // namespace

//===------------------------------------------------------------------===//
// VerifDialect
//===------------------------------------------------------------------===//

void llzk::verif::attachInterfaces(MLIRContext &context) {
  scf::WhileOp::attachInterface<ScfWhileExternalModel>(context);
  scf::ForOp::attachInterface<ScfForExternalModel>(context);
}

void llzk::verif::registerExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(
      TypeID::get<InterfacesExtension>(), std::make_unique<InterfacesExtension>()
  );
}

auto llzk::verif::VerifDialect::initialize() -> void {
  // clang-format off
  addOperations<
    #define GET_OP_LIST
    #include "llzk/Dialect/Verif/IR/Ops.cpp.inc"
  >();
  // clang-format on
  addInterfaces<LLZKDialectBytecodeInterface<VerifDialect>>();
  declarePromisedInterface<llzk::verif::InvariantTargetOpInterface, scf::WhileOp>();
  declarePromisedInterface<llzk::verif::InvariantTargetOpInterface, scf::ForOp>();
}
