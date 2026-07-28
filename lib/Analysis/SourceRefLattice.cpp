//===-- SourceRefLattice.cpp - SourceRef lattice & utils --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SourceRefLattice.h"

#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/POD/IR/Ops.h"
#include "llzk/Util/Hash.h"
#include "llzk/Util/SymbolHelper.h"

#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/IR/Value.h>

#include <llvm/Support/Debug.h>

#include <numeric>
#include <unordered_set>

#define DEBUG_TYPE "llzk-constrain-ref-lattice"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace felt;
using namespace pod;
using namespace polymorphic;

/* SourceRefLatticeValue */

mlir::ChangeResult SourceRefLatticeValue::insert(const SourceRef &rhs) {
  auto rhsVal = SourceRefLatticeValue(rhs);
  if (isScalar()) {
    return updateScalar(rhsVal.getScalarValue());
  } else {
    return foldAndUpdate(rhsVal);
  }
}

std::pair<SourceRefLatticeValue, mlir::ChangeResult>
SourceRefLatticeValue::translate(const TranslationMap &translation) const {
  auto newVal = *this;
  auto res = mlir::ChangeResult::NoChange;
  if (newVal.isScalar()) {
    res = newVal.translateScalar(translation);
  } else {
    for (auto &elem : newVal.getArrayValue()) {
      auto [newElem, elemRes] = elem->translate(translation);
      (*elem) = newElem;
      res |= elemRes;
    }
  }
  return {newVal, res};
}

mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
SourceRefLatticeValue::referenceMember(SymbolLookupResult<MemberDefOp> memberRef) const {
  SourceRefIndex idx(std::move(memberRef));
  auto transform = [&idx](const SourceRef &r) -> mlir::FailureOr<SourceRef> {
    return r.createChild(idx);
  };
  return elementwiseTransform(transform);
}

mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
SourceRefLatticeValue::referencePodRecord(mlir::StringAttr recordName) const {
  SourceRefIndex idx(recordName);
  auto transform = [&idx](const SourceRef &r) -> mlir::FailureOr<SourceRef> {
    return r.createChild(idx);
  };
  return elementwiseTransform(transform);
}

mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
SourceRefLatticeValue::extract(const std::vector<SourceRefIndex> &indices) const {
  if (isArray()) {
    ensure(indices.size() <= getNumArrayDims(), "invalid extract array operands");

    // First, compute what chunk(s) to index
    std::vector<size_t> currIdxs {0};
    for (unsigned i = 0; i < indices.size(); i++) {
      const auto &idx = indices[i];
      auto currDim = getArrayDim(i);

      std::vector<size_t> newIdxs;
      ensure(idx.isIndex() || idx.isIndexRange(), "wrong type of index for array");
      if (idx.isIndex()) {
        int64_t idxVal(idx.getIndex());
        std::transform(
            currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
            [&currDim, &idxVal](size_t j) { return j * currDim + idxVal; }
        );
      } else {
        auto [low, high] = idx.getIndexRange();
        int64_t lowInt(low), highInt(high);
        for (int64_t idxVal = lowInt; idxVal < highInt; idxVal++) {
          std::transform(
              currIdxs.begin(), currIdxs.end(), std::back_inserter(newIdxs),
              [&currDim, &idxVal](size_t j) { return j * currDim + idxVal; }
          );
        }
      }

      currIdxs = newIdxs;
    }
    std::vector<int64_t> newArrayDims;
    size_t chunkSz = 1;
    for (size_t i = indices.size(); i < getNumArrayDims(); i++) {
      auto dim = getArrayDim(i);
      newArrayDims.push_back(dim);
      chunkSz *= dim;
    }
    if (newArrayDims.empty()) {
      // read case, where the return value is a scalar (single element)
      SourceRefLatticeValue extractedVal;
      for (auto idx : currIdxs) {
        (void)extractedVal.update(getElemFlatIdx(idx));
      }
      return std::make_pair(extractedVal, mlir::ChangeResult::Change);
    } else {
      // extract case, where the return value is an array of fewer dimensions.
      SourceRefLatticeValue extractedVal(newArrayDims);
      for (auto chunkStart : currIdxs) {
        for (size_t i = 0; i < chunkSz; i++) {
          (void)extractedVal.getElemFlatIdx(i).update(getElemFlatIdx(chunkStart + i));
        }
      }
      return std::make_pair(extractedVal, mlir::ChangeResult::Change);
    }
  } else {
    auto currVal = *this;
    auto res = mlir::ChangeResult::NoChange;
    for (const auto &idx : indices) {
      auto transform = [&idx](const SourceRef &r) -> mlir::FailureOr<SourceRef> {
        return r.createChild(idx);
      };
      auto transformedVal = currVal.elementwiseTransform(transform);
      if (failed(transformedVal)) {
        return mlir::failure();
      }
      auto [newVal, transformRes] = *transformedVal;
      currVal = std::move(newVal);
      res |= transformRes;
    }
    return std::make_pair(currVal, res);
  }
}

mlir::ChangeResult SourceRefLatticeValue::translateScalar(const TranslationMap &translation) {
  auto res = mlir::ChangeResult::NoChange;
  // copy the current value
  auto currVal = getScalarValue();
  // reset this value
  getValue() = ScalarTy();
  // For each current element, see if the translation map contains a valid prefix.
  // If so, translate the current element with all replacement prefixes indicated
  // by the translation value.
  for (const SourceRef &currRef : currVal) {
    for (const auto &[prefix, replacementVal] : translation) {
      if (currRef.isValidPrefix(prefix)) {
        for (const SourceRef &replacementPrefix : replacementVal.foldToScalar()) {
          auto translatedRefRes = currRef.translate(prefix, replacementPrefix);
          if (succeeded(translatedRefRes)) {
            res |= insert(*translatedRefRes);
          }
        }
      }
    }
  }
  return res;
}

mlir::FailureOr<std::pair<SourceRefLatticeValue, mlir::ChangeResult>>
SourceRefLatticeValue::elementwiseTransform(
    llvm::function_ref<mlir::FailureOr<SourceRef>(const SourceRef &)> transform
) const {
  auto newVal = *this;
  auto res = mlir::ChangeResult::NoChange;
  if (newVal.isScalar()) {
    ScalarTy indexed;
    for (const auto &ref : newVal.getScalarValue()) {
      auto transformedRef = transform(ref);
      if (failed(transformedRef)) {
        return mlir::failure();
      }
      auto [_, inserted] = indexed.insert(*transformedRef);
      if (inserted) {
        res |= mlir::ChangeResult::Change;
      }
    }
    newVal.getScalarValue() = indexed;
  } else {
    for (auto &elem : newVal.getArrayValue()) {
      auto transformedElem = elem->elementwiseTransform(transform);
      if (failed(transformedElem)) {
        return mlir::failure();
      }
      auto [newElem, elemRes] = *transformedElem;
      (*elem) = std::move(newElem);
      res |= elemRes;
    }
  }
  return std::make_pair(newVal, res);
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRefLatticeValue &v) {
  v.print(os);
  return os;
}

/* SourceRefLattice */

mlir::FailureOr<SourceRef> SourceRefLattice::getSourceRef(mlir::Value val) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
    return SourceRef(blockArg);
  } else if (auto *defOp = val.getDefiningOp()) {
    if (auto feltConst = llvm::dyn_cast<FeltConstantOp>(defOp)) {
      return SourceRef(feltConst);
    } else if (auto constIdx = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(defOp)) {
      return SourceRef(constIdx);
    } else if (auto readConst = llvm::dyn_cast<ConstReadOp>(defOp)) {
      return SourceRef(readConst);
    } else if (auto structNew = llvm::dyn_cast<CreateStructOp>(defOp)) {
      return SourceRef(structNew);
    } else if (auto nonDet = llvm::dyn_cast<NonDetOp>(defOp)) {
      return SourceRef(nonDet);
    } else if (auto createArray = llvm::dyn_cast<CreateArrayOp>(defOp)) {
      return SourceRef(createArray->getResult(0));
    } else if (auto newPod = llvm::dyn_cast<NewPodOp>(defOp)) {
      return SourceRef(newPod->getResult(0));
    } else if (llvm::isa<function::CallOp>(defOp)) {
      auto callResult = llvm::dyn_cast<mlir::OpResult>(val);
      ensure(callResult != nullptr, "function.call value should be an OpResult");
      return SourceRef(callResult);
    }
  }
  return mlir::failure();
}

SourceRefLatticeValue SourceRefLattice::getDefaultValue(SourceRefLattice::ValueTy v) {
  if (auto asVal = llvm::dyn_cast_if_present<Value>(v)) {
    auto sourceRef = getSourceRef(asVal);
    if (mlir::succeeded(sourceRef)) {
      return SourceRefLatticeValue(*sourceRef);
    }
  }
  return SourceRefLatticeValue();
}

ChangeResult SourceRefLattice::join(const AbstractSparseLattice &rhs) {
  return value.update(static_cast<const SourceRefLattice &>(rhs).value);
}

ChangeResult SourceRefLattice::meet(const AbstractSparseLattice & /*rhs*/) {
  llvm::report_fatal_error("meet operation is not supported for SourceRefLattice");
  return ChangeResult::NoChange;
}

void SourceRefLattice::print(mlir::raw_ostream &os) const {
  os << "SourceRefLattice { " << value << " }";
}

ChangeResult SourceRefLattice::setValue(const LatticeValue &newValue) {
  return value.setValue(newValue);
}

ChangeResult SourceRefLattice::setValue(const SourceRef &ref) {
  return value.setValue(LatticeValue(ref));
}

} // namespace llzk

namespace llvm {

raw_ostream &operator<<(raw_ostream &os, llvm::PointerUnion<mlir::Value, mlir::Operation *> ptr) {
  if (auto asVal = llvm::dyn_cast_if_present<Value>(ptr)) {
    os << asVal;
  } else if (auto *asOp = llvm::dyn_cast_if_present<Operation *>(ptr)) {
    os << *asOp;
  } else {
    os << "<<null PointerUnion>>";
  }
  return os;
}
} // namespace llvm
