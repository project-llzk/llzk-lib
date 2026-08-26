//===-- PCLLoweringPass.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2026 Project LLZK
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-to-pcl` pass.
///
//===----------------------------------------------------------------------===//

#include "PCLLoweringPass/Modes.h"
#include "pcl/Conversion/ConversionPasses.h"

#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/Field.h"

#include <pcl/Dialect/IR/Dialect.h>
#include <pcl/Dialect/IR/Ops.h>
#include <pcl/Dialect/IR/Types.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/IRMapping.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseMapInfo.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/EquivalenceClasses.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>

#include <memory>
#include <optional>

// Include the generated base pass class definitions.
namespace pcl {

#define GEN_PASS_DEF_PCLLOWERINGPASS
#include "pcl/Conversion/ConversionPasses.h.inc"
} // namespace pcl

using namespace mlir;
using namespace llzk;
using namespace llzk::felt;
using namespace llzk::component;

namespace {

class PassImpl : public pcl::impl::PCLLoweringPassBase<PassImpl> {
  using Base = PCLLoweringPassBase<PassImpl>;
  using Base::Base;

  /// The translation only works now on LLZK structs where all the members are felts or
  /// subcomponents.
  LogicalResult validateStructs() {
    return failure(
        getOperation()
            ->walk([](StructDefOp op) -> WalkResult {
      for (auto member : op.getMemberDefs()) {
        auto memberType = member.getType();
        if (!llvm::isa<FeltType, StructType>(memberType)) {
          return member.emitError() << "Member must be felt or struct type. Found " << memberType
                                    << " for member: " << member.getName();
        }
      }
      return success();
    }).wasInterrupted()
    );
  }

  // PCL programs require a module-level attribute specifying the prime.
  void setPrime(APInt &prime) {
    Operation *op = getOperation();
    op->setAttrs(
        DictionaryAttr::get(
            &getContext(),
            {NamedAttribute(PCL_PRIME_ATTR_NAME, pcl::PrimeAttr::get(&getContext(), prime))}
        )
    );
  }

  FailureOr<APSInt> selectPrime() {
    Operation *op = getOperation();
    FieldSet fields;
    // If the collection reports that at least one FeltType did not declare the field and
    // the fields set is empty, then we raise an error.
    if (failed(collectFields(op, fields)) && fields.empty()) {
      return op->emitOpError() << "could not deduce the prime field";
    }
    // If the fields is empty and we reached this point it means that the IR we are about to lower
    // does not have a single felt type (because felts without a field will make `collectFields`
    // return failure). We return an error here since we don't have a prime to emit. In practice,
    // this situation it's going to be unlikely.
    if (fields.empty()) {
      return op->emitOpError() << "does not contain felt types and prime field couldn't be deduced";
    }
    // The pass only supports having one field for the whole circuit.
    if (fields.size() > 1) {
      return op->emitOpError() << "multiple fields is not supported";
    }
    const auto &selectedField = *(fields.begin());
    return toAPSInt(selectedField.get().prime());
  }

  std::unique_ptr<pcl::lowering::BaseMode> createMode() {
    switch (mode.getValue()) {
    case pcl::LlzkToPclMode::Full:
      return std::make_unique<pcl::lowering::FullLoweringMode>(getOperation());
    case pcl::LlzkToPclMode::Stubbed:
      return std::make_unique<pcl::lowering::StubbedLoweringMode>(getOperation());
    }
    llvm_unreachable("only two lowering modes");
  }

  void runOnOperation() override {
    auto prime = selectPrime();
    if (failed(prime) || failed(validateStructs()) || failed(createMode()->lower())) {
      signalPassFailure();
      return;
    }

    setPrime(*prime);
  }
};
} // namespace
