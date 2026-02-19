//===-- SourceRefLattice.cpp - SourceRef lattice & utils --*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/ConstraintDependencyGraph.h"
#include "llzk/Analysis/SourceRefLattice.h"
#include "llzk/Dialect/Felt/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
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

using namespace component;
using namespace felt;
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

std::pair<SourceRefLatticeValue, mlir::ChangeResult>
SourceRefLatticeValue::referenceMember(SymbolLookupResult<MemberDefOp> memberRef) const {
  SourceRefIndex idx(memberRef);
  auto transform = [&idx](const SourceRef &r) -> SourceRef { return r.createChild(idx); };
  return elementwiseTransform(transform);
}

std::pair<SourceRefLatticeValue, mlir::ChangeResult>
SourceRefLatticeValue::extract(const std::vector<SourceRefIndex> &indices) const {
  if (isArray()) {
    ensure(indices.size() <= getNumArrayDims(), "invalid extract array operands");

    // First, compute what chunk(s) to index
    std::vector<size_t> currIdxs {0};
    for (unsigned i = 0; i < indices.size(); i++) {
      auto &idx = indices[i];
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
    for (unsigned i = indices.size(); i < getNumArrayDims(); i++) {
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
      return {extractedVal, mlir::ChangeResult::Change};
    } else {
      // extract case, where the return value is an array of fewer dimensions.
      SourceRefLatticeValue extractedVal(newArrayDims);
      for (auto chunkStart : currIdxs) {
        for (size_t i = 0; i < chunkSz; i++) {
          (void)extractedVal.getElemFlatIdx(i).update(getElemFlatIdx(chunkStart + i));
        }
      }
      return {extractedVal, mlir::ChangeResult::Change};
    }
  } else {
    auto currVal = *this;
    auto res = mlir::ChangeResult::NoChange;
    for (auto &idx : indices) {
      auto transform = [&idx](const SourceRef &r) -> SourceRef { return r.createChild(idx); };
      auto [newVal, transformRes] = currVal.elementwiseTransform(transform);
      currVal = std::move(newVal);
      res |= transformRes;
    }
    return {currVal, res};
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
    for (auto &[prefix, replacementVal] : translation) {
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

std::pair<SourceRefLatticeValue, mlir::ChangeResult> SourceRefLatticeValue::elementwiseTransform(
    llvm::function_ref<SourceRef(const SourceRef &)> transform
) const {
  auto newVal = *this;
  auto res = mlir::ChangeResult::NoChange;
  if (newVal.isScalar()) {
    ScalarTy indexed;
    for (auto &ref : newVal.getScalarValue()) {
      auto [_, inserted] = indexed.insert(transform(ref));
      if (inserted) {
        res |= mlir::ChangeResult::Change;
      }
    }
    newVal.getScalarValue() = indexed;
  } else {
    for (auto &elem : newVal.getArrayValue()) {
      auto [newElem, elemRes] = elem->elementwiseTransform(transform);
      (*elem) = newElem;
      res |= elemRes;
    }
  }
  return {newVal, res};
}

mlir::raw_ostream &operator<<(mlir::raw_ostream &os, const SourceRefLatticeValue &v) {
  v.print(os);
  return os;
}

/* SourceRefLattice */

mlir::FailureOr<SourceRef> SourceRefLattice::getSourceRef(mlir::Value val) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(val)) {
    return SourceRef(blockArg);
  } else if (auto defOp = val.getDefiningOp()) {
    if (auto feltConst = llvm::dyn_cast<FeltConstantOp>(defOp)) {
      return SourceRef(feltConst);
    } else if (auto constIdx = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(defOp)) {
      return SourceRef(constIdx);
    } else if (auto readConst = llvm::dyn_cast<ConstReadOp>(defOp)) {
      return SourceRef(readConst);
    } else if (auto structNew = llvm::dyn_cast<CreateStructOp>(defOp)) {
      return SourceRef(structNew);
    }
  }
  return mlir::failure();
}

void SourceRefLattice::print(mlir::raw_ostream &os) const {
  os << "SourceRefLattice { ";
  for (auto mit = valMap.begin(); mit != valMap.end();) {
    auto &[val, latticeVal] = *mit;
    os << "\n    (";
    if (auto asVal = llvm::dyn_cast<Value>(val)) {
      os << asVal;
    } else if (auto asOp = llvm::dyn_cast<Operation *>(val)) {
      os << *asOp;
    } else {
      llvm_unreachable("unhandled ValueTy print case");
    }
    os << ") => " << latticeVal;
    mit++;
    if (mit != valMap.end()) {
      os << ',';
    } else {
      os << '\n';
    }
  }
  os << "}\n";
}

mlir::ChangeResult SourceRefLattice::setValues(const ValueMap &rhs) {
  auto res = mlir::ChangeResult::NoChange;
  for (auto &[v, s] : rhs) {
    res |= setValue(v, s);
  }
  return res;
}

mlir::ChangeResult SourceRefLattice::setValue(ValueTy v, const SourceRefLatticeValue &rhs) {
  for (const SourceRef &ref : rhs.foldToScalar()) {
    refMap[ref].insert(v);
  }
  return valMap[v].setValue(rhs);
}

mlir::ChangeResult SourceRefLattice::setValue(ValueTy v, const SourceRef &ref) {
  refMap[ref].insert(v);
  return valMap[v].setValue(SourceRefLatticeValue(ref));
}

SourceRefLatticeValue SourceRefLattice::getOrDefault(SourceRefLattice::ValueTy v) const {
  auto it = valMap.find(v);
  if (it != valMap.end()) {
    return it->second;
  }

  if (auto asVal = llvm::dyn_cast_if_present<Value>(v)) {
    auto sourceRef = getSourceRef(asVal);
    if (mlir::succeeded(sourceRef)) {
      return SourceRefLatticeValue(sourceRef.value());
    }
  }
  return SourceRefLatticeValue();
}

SourceRefLatticeValue SourceRefLattice::getReturnValue(unsigned i) const {
  ProgramPoint *pp = llvm::cast<ProgramPoint *>(this->getAnchor());
  if (auto retOp = mlir::dyn_cast_if_present<function::ReturnOp>(pp->getPrevOp())) {
    if (i >= retOp.getNumOperands()) {
      llvm::report_fatal_error("return value requested is out of range");
    }
    return this->getOrDefault(retOp.getOperand(i));
  }
  return SourceRefLatticeValue();
}

SourceRefLattice::ValueSet SourceRefLattice::lookupValues(const SourceRef &ref) const {
  if (auto it = refMap.find(ref); it != refMap.end()) {
    return it->second;
  }
  return SourceRefLattice::ValueSet();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const SourceRefLattice &lattice) {
  lattice.print(os);
  return os;
}

} // namespace llzk

namespace llvm {

raw_ostream &operator<<(raw_ostream &os, llvm::PointerUnion<mlir::Value, mlir::Operation *> ptr) {
  if (auto asVal = llvm::dyn_cast_if_present<Value>(ptr)) {
    os << asVal;
  } else if (auto asOp = llvm::dyn_cast_if_present<Operation *>(ptr)) {
    os << *asOp;
  } else {
    os << "<<null PointerUnion>>";
  }
  return os;
}
} // namespace llvm
