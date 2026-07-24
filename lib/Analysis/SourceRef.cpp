//===-- SourceRef.cpp - SourceRef implementation ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "llzk/Analysis/SourceRef.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/String/IR/Types.h"
#include "llzk/Transforms/LLZKLoweringUtils.h"
#include "llzk/Util/Compare.h"
#include "llzk/Util/Debug.h"
#include "llzk/Util/DynamicAPIntHelper.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/AsmState.h>

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace felt;
using namespace function;
using namespace polymorphic;
using namespace string;

namespace {

std::strong_ordering
compareDynamicAPInt(const llvm::DynamicAPInt &lhs, const llvm::DynamicAPInt &rhs) {
  if (lhs < rhs) {
    return std::strong_ordering::less;
  }
  if (rhs < lhs) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

std::strong_ordering compareStringRef(llvm::StringRef lhs, llvm::StringRef rhs) {
  int cmp = lhs.compare(rhs);
  if (cmp < 0) {
    return std::strong_ordering::less;
  }
  if (cmp > 0) {
    return std::strong_ordering::greater;
  }
  return std::strong_ordering::equal;
}

// Prints SourceRef path using source-style syntax, i.e. `.` for struct members
// and pod records and `[...]` for array indices.
// Returns failure if an unexpected `SourceRefIndex` type is encountered.
LogicalResult printSourceStylePath(raw_ostream &os, llvm::ArrayRef<SourceRefIndex> path) {
  for (const auto &idx : path) {
    if (idx.isMember()) {
      os << '.' << idx.getMember().getName();
      continue;
    }
    if (idx.isPodRecord()) {
      os << '.' << idx.getPodRecordName();
      continue;
    }
    if (idx.isIndex() || idx.isIndexRange()) {
      os << '[' << idx << ']';
      continue;
    }
    return failure();
  }
  return success();
}

std::strong_ordering
compareSourceRefPaths(llvm::ArrayRef<SourceRefIndex> lhs, llvm::ArrayRef<SourceRefIndex> rhs) {
  for (size_t i = 0; i < lhs.size() && i < rhs.size(); i++) {
    if (auto cmp = lhs[i] <=> rhs[i]; cmp != std::strong_ordering::equal) {
      return cmp;
    }
  }
  return lhs.size() <=> rhs.size();
}

} // namespace

/* SourceRefIndex */

void SourceRefIndex::print(raw_ostream &os) const {
  if (isMember()) {
    os << '@' << getMember().getName();
  } else if (isPodRecord()) {
    os << '@' << getPodRecordName();
  } else if (isIndex()) {
    os << getIndex();
  } else {
    auto [low, high] = getIndexRange();
    if (ShapedType::isDynamic(int64_t(high))) {
      os << "<dynamic>";
    } else {
      os << low << ':' << high;
    }
  }
}

std::strong_ordering SourceRefIndex::operator<=>(const SourceRefIndex &rhs) const {
  if (isMember() && rhs.isMember()) {
    if (NamedOpLocationLess<MemberDefOp> {}(getMember(), rhs.getMember())) {
      return std::strong_ordering::less;
    }
    if (NamedOpLocationLess<MemberDefOp> {}(rhs.getMember(), getMember())) {
      return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
  }
  if (isPodRecord() && rhs.isPodRecord()) {
    return compareStringRef(getPodRecordName(), rhs.getPodRecordName());
  }
  if (isIndex() && rhs.isIndex()) {
    return compareDynamicAPInt(getIndex(), rhs.getIndex());
  }
  if (isIndexRange() && rhs.isIndexRange()) {
    auto [ll, lu] = getIndexRange();
    auto [rl, ru] = rhs.getIndexRange();
    if (auto cmp = compareDynamicAPInt(ll, rl); cmp != std::strong_ordering::equal) {
      return cmp;
    }
    return compareDynamicAPInt(lu, ru);
  }

  if (isMember()) {
    return std::strong_ordering::less;
  }
  if (rhs.isMember()) {
    return std::strong_ordering::greater;
  }
  if (isPodRecord()) {
    return std::strong_ordering::less;
  }
  if (rhs.isPodRecord()) {
    return std::strong_ordering::greater;
  }
  if (isIndex()) {
    return std::strong_ordering::less;
  }
  return std::strong_ordering::greater;
}

size_t SourceRefIndex::Hash::operator()(const SourceRefIndex &c) const {
  if (c.isIndex()) {
    // We don't hash the index directly, because the built-in LLVM hash includes
    // the bitwidth of the APInt in the hash, which is undesirable for this application.
    // i.e., We want a N-bit version of x to hash to the same value as an M-bit version of X,
    // because our equality checks would consider them equal regardless of bitwidth.
    APSInt idx = toAPSInt(c.getIndex());
    unsigned requiredBits = idx.getSignificantBits();
    auto hash = llvm::hash_value(idx.trunc(requiredBits));
    return hash;
  } else if (c.isIndexRange()) {
    auto r = c.getIndexRange();
    return llvm::hash_value(std::get<0>(r)) ^ llvm::hash_value(std::get<1>(r));
  } else if (c.isPodRecord()) {
    return llvm::hash_value(c.getPodRecordName());
  } else {
    return OpHash<component::MemberDefOp> {}(c.getMember());
  }
}

bool SourceRefIndex::overlaps(const SourceRefIndex &rhs) const {
  if (isIndex() && rhs.isIndexRange()) {
    auto [low, high] = rhs.getIndexRange();
    return low <= getIndex() && getIndex() < high;
  }
  if (isIndexRange() && rhs.isIndex()) {
    return rhs.overlaps(*this);
  }
  if (isIndexRange() && rhs.isIndexRange()) {
    auto [lhsLow, lhsHigh] = getIndexRange();
    auto [rhsLow, rhsHigh] = rhs.getIndexRange();
    return lhsLow < rhsHigh && rhsLow < lhsHigh;
  }
  return *this == rhs;
}

/* SourceRef */

SourceRef::SortCategory SourceRef::getSortCategory() const {
  if (isBlockArgument()) {
    return SortCategory::BlockArgument;
  }
  if (isCreateStructOp()) {
    return SortCategory::CreateStruct;
  }
  if (isNonDetOp()) {
    return SortCategory::NonDet;
  }
  if (isRooted()) {
    return SortCategory::RootResult;
  }
  if (isTemplateConstant()) {
    return SortCategory::TemplateConstant;
  }
  if (isConstantIndex()) {
    return SortCategory::ConstantIndex;
  }
  if (isConstantFelt()) {
    return SortCategory::ConstantFelt;
  }

  llvm::errs() << *this << '\n';
  llvm_unreachable("unhandled SourceRef sort category");
}

StringRef SourceRef::getTemplateConstantName() const {
  auto constantVal = getConstant();
  ensure(succeeded(constantVal), "template constant must be constant");
  auto constRead = llvm::dyn_cast<ConstReadOp>(constantVal->getDefiningOp());
  ensure(constRead, "template constant must be backed by const.read");
  return constRead.getConstName();
}

std::strong_ordering
SourceRef::compareWithinCategory(const SourceRef &rhs, SortCategory category) const {
  switch (category) {
  case SortCategory::BlockArgument: {
    if (auto cmp = *getInputNum() <=> *rhs.getInputNum(); cmp != std::strong_ordering::equal) {
      return cmp;
    }
    if (auto cmp = getAsOpaquePointer() <=> rhs.getAsOpaquePointer();
        cmp != std::strong_ordering::equal) {
      return cmp;
    }
    return compareSourceRefPaths(getPath(), rhs.getPath());
  }
  case SortCategory::CreateStruct:
  case SortCategory::NonDet:
  case SortCategory::RootResult: {
    if (auto cmp = getAsOpaquePointer() <=> rhs.getAsOpaquePointer();
        cmp != std::strong_ordering::equal) {
      return cmp;
    }
    return compareSourceRefPaths(getPath(), rhs.getPath());
  }
  case SortCategory::TemplateConstant: {
    if (auto cmp = compareStringRef(getTemplateConstantName(), rhs.getTemplateConstantName());
        cmp != std::strong_ordering::equal) {
      return cmp;
    }
    return getAsOpaquePointer() <=> rhs.getAsOpaquePointer();
  }
  case SortCategory::ConstantIndex:
    return compareDynamicAPInt(*getConstantIndexValue(), *rhs.getConstantIndexValue());
  case SortCategory::ConstantFelt:
    return compareDynamicAPInt(*getConstantFeltValue(), *rhs.getConstantFeltValue());
  }

  llvm_unreachable("unhandled SourceRef category compare");
}

/// @brief Lookup a `StructDefOp` from a given `StructType`.
/// @param tables
/// @param mod
/// @param ty
/// @return A `SymbolLookupResult` for the `StructDefOp` found. Note that returning the
/// lookup result is important, as it may manage a ModuleOp if the struct is found
/// via an include.
SymbolLookupResult<StructDefOp>
getStructDef(SymbolTableCollection &tables, ModuleOp mod, StructType ty) {
  auto sDef = ty.getDefinition(tables, mod);
  ensure(
      succeeded(sDef),
      "could not find '" + StructDefOp::getOperationName() + "' op from struct type"
  );

  return std::move(*sDef);
}

std::vector<SourceRef>
SourceRef::getAllSourceRefs(SymbolTableCollection &tables, ModuleOp mod, const SourceRef &root) {
  std::vector<SourceRef> res = {root};
  for (const SourceRef &child : root.getAllChildren(tables, mod)) {
    auto recursiveChildren = getAllSourceRefs(tables, mod, child);
    res.insert(res.end(), recursiveChildren.begin(), recursiveChildren.end());
  }
  return res;
}

std::vector<SourceRef> SourceRef::getAllSourceRefs(StructDefOp structDef, FuncDefOp fnOp) {
  std::vector<SourceRef> res;

  ensure(
      structDef == fnOp->getParentOfType<StructDefOp>(), "function must be within the given struct"
  );

  FailureOr<ModuleOp> modOp = getRootModule(structDef);
  ensure(succeeded(modOp), "could not lookup module from struct " + Twine(structDef.getName()));

  SymbolTableCollection tables;
  for (auto a : fnOp.getArguments()) {
    auto argRes = getAllSourceRefs(tables, modOp.value(), SourceRef(a));
    res.insert(res.end(), argRes.begin(), argRes.end());
  }

  // For compute functions, the "self" member is not arg0 like for constrain, but
  // rather the struct value returned from the function.
  if (fnOp.isStructCompute()) {
    Value selfVal = fnOp.getSelfValueFromCompute();
    auto createOp = dyn_cast_if_present<CreateStructOp>(selfVal.getDefiningOp());
    ensure(createOp, "self value should originate from struct.new operation");
    auto selfRes = getAllSourceRefs(tables, modOp.value(), SourceRef(createOp));
    res.insert(res.end(), selfRes.begin(), selfRes.end());
  }

  return res;
}

std::vector<SourceRef> SourceRef::getAllSourceRefs(StructDefOp structDef, MemberDefOp memberDef) {
  std::vector<SourceRef> res;
  FuncDefOp constrainFnOp = structDef.getConstrainFuncOp();
  ensure(
      memberDef->getParentOfType<StructDefOp>() == structDef,
      "Member " + Twine(memberDef.getName()) + " is not a member of struct " +
          Twine(structDef.getName())
  );
  FailureOr<ModuleOp> modOp = getRootModule(structDef);
  ensure(succeeded(modOp), "could not lookup module from struct " + Twine(structDef.getName()));

  // Get the self argument (like `FuncDefOp::getSelfValueFromConstrain()`)
  BlockArgument self = constrainFnOp.getArguments().front();
  SourceRef memberRef = SourceRef(self, {SourceRefIndex(memberDef)});

  SymbolTableCollection tables;
  return getAllSourceRefs(tables, modOp.value(), memberRef);
}

Type SourceRef::getType() const {
  Type currTy = value.getType();
  for (const auto &idx : getPath()) {
    if (idx.isMember()) {
      currTy = idx.getMember().getType();
      continue;
    }
    if (idx.isPodRecord()) {
      auto podTy = dyn_cast<pod::PodType>(currTy);
      ensure(static_cast<bool>(podTy), "SourceRef pod record requires a pod-typed base");
      auto lookup = podTy.getRecord(idx.getPodRecordName(), [ctx = value.getContext()]() {
        return mlir::emitError(
            mlir::UnknownLoc::get(ctx), "SourceRef references a missing pod record"
        );
      });
      ensure(succeeded(lookup), "SourceRef references a missing pod record");
      currTy = *lookup;
      continue;
    }

    auto arrTy = dyn_cast<ArrayType>(currTy);
    ensure(static_cast<bool>(arrTy), "SourceRef array index requires an array-typed base");
    currTy = arrTy.getSelectionType(1);
  }
  return currTy;
}

bool SourceRef::isValidPrefix(const SourceRef &prefix) const {
  if (isConstant() || prefix.isConstant()) {
    return false;
  }

  auto pathRef = getPath();
  auto prefixPath = prefix.getPath();
  if (value != prefix.value || pathRef.size() < prefixPath.size()) {
    return false;
  }
  for (size_t i = 0; i < prefixPath.size(); i++) {
    if (pathRef[i] != prefixPath[i]) {
      return false;
    }
  }
  return true;
}

bool SourceRef::overlaps(const SourceRef &rhs) const {
  if (isConstant() || rhs.isConstant() || value != rhs.value || path.size() != rhs.path.size()) {
    return false;
  }
  return llvm::all_of(llvm::zip(path, rhs.path), [](const auto &indices) {
    return std::get<0>(indices).overlaps(std::get<1>(indices));
  });
}

FailureOr<SourceRef::Path> SourceRef::getSuffix(const SourceRef &prefix) const {
  if (!isValidPrefix(prefix)) {
    return failure();
  }
  Path suffix;
  auto pathRef = getPath();
  auto prefixPath = prefix.getPath();
  suffix.reserve(pathRef.size() - prefixPath.size());
  for (size_t i = prefixPath.size(); i < pathRef.size(); i++) {
    suffix.push_back(pathRef[i]);
  }
  return suffix;
}

FailureOr<SourceRef> SourceRef::translate(const SourceRef &prefix, const SourceRef &other) const {
  if (isConstant()) {
    return *this;
  }
  auto suffix = getSuffix(prefix);
  if (failed(suffix)) {
    return failure();
  }

  SourceRef newSignalUsage = other; // copy
  if (newSignalUsage.isRooted()) {
    SourceRef::Path &pathRef = newSignalUsage.getPathMut();
    pathRef.insert(pathRef.end(), suffix->begin(), suffix->end());
  }

  return newSignalUsage;
}

std::vector<SourceRef> getAllChildren(
    SymbolTableCollection & /*tables*/, ModuleOp /*mod*/, ArrayType arrayTy, const SourceRef &root
) {
  std::vector<SourceRef> res;
  // Recurse into arrays by iterating over their elements
  for (int64_t i = 0; i < arrayTy.getDimSize(0); i++) {
    auto childRef = root.createChild(SourceRefIndex(i));
    ensure(succeeded(childRef), "array children require a rooted SourceRef");
    res.push_back(*childRef);
  }

  return res;
}

std::vector<SourceRef> getAllChildren(
    SymbolTableCollection &tables, ModuleOp mod, SymbolLookupResult<StructDefOp> structDefRes,
    const SourceRef &root
) {
  std::vector<SourceRef> res;
  // Recurse into struct types by iterating over all their member definitions
  for (auto f : structDefRes.get().getOps<MemberDefOp>()) {
    // We want to store the MemberDefOp, but without the possibility of accidentally dropping the
    // reference, so we need to re-lookup the symbol to create a SymbolLookupResult, which will
    // manage the external module containing the member defs, if needed.
    // TODO: It would be nice if we could manage module op references differently
    // so we don't have to do this.
    auto structDefCopy = structDefRes;
    auto memberLookup = lookupSymbolIn<MemberDefOp>(
        tables, SymbolRefAttr::get(f.getContext(), f.getSymNameAttr()), std::move(structDefCopy),
        mod.getOperation()
    );
    ensure(succeeded(memberLookup), "could not get SymbolLookupResult of existing MemberDefOp");
    auto childRef = root.createChild(SourceRefIndex(memberLookup.value()));
    ensure(succeeded(childRef), "struct children require a rooted SourceRef");
    // Make a reference to the current member, regardless of if it is a composite
    // type or not.
    res.push_back(*childRef);
  }
  return res;
}

std::vector<SourceRef> getAllChildren(pod::PodType podTy, const SourceRef &root) {
  std::vector<SourceRef> res;
  for (auto record : podTy.getRecords()) {
    auto childRef = root.createChild(SourceRefIndex(record.getName()));
    ensure(succeeded(childRef), "pod children require a rooted SourceRef");
    res.push_back(*childRef);
  }
  return res;
}

std::vector<SourceRef>
SourceRef::getAllChildren(SymbolTableCollection &tables, ModuleOp mod) const {
  auto ty = getType();
  if (auto structTy = dyn_cast<StructType>(ty)) {
    return llzk::getAllChildren(tables, mod, getStructDef(tables, mod, structTy), *this);
  } else if (auto podTy = dyn_cast<pod::PodType>(ty)) {
    return llzk::getAllChildren(podTy, *this);
  } else if (auto arrayType = dyn_cast<ArrayType>(ty)) {
    return llzk::getAllChildren(tables, mod, arrayType, *this);
  }
  // Scalar type, no children
  return {};
}

static void printCallResultFallback(raw_ostream &os, function::CallOp callOp, Value value) {
  os << "<call " << callOp.getCallee();
  os << ' ';
  Operation *printScope = callOp.getOperation();
  if (auto funcOp = callOp->getParentOfType<FuncDefOp>()) {
    printScope = funcOp.getOperation();
  }
  // Allows us to print the SSA result value of the call to disambiguate
  // repeated calls in the same function.
  AsmState state(printScope);
  value.printAsOperand(os, state);
  os << '>';
}

static bool shouldPrintNamedCallResult(
    function::CallOp callOp, OpResult callResult, function::FuncDefOp calleeFunc
) {
  auto resName = calleeFunc.getResNameAttr(callResult.getResultNumber());
  if (!resName) {
    return false;
  }

  auto parentFunc = callOp->getParentOfType<FuncDefOp>();
  if (!parentFunc) {
    return true;
  }

  bool foundThisCall = false;
  bool foundDuplicate = false;
  parentFunc.walk([&](function::CallOp otherCall) {
    if (foundDuplicate) {
      return WalkResult::interrupt();
    }

    auto otherFunc = llvm::dyn_cast_if_present<FuncDefOp>(otherCall.resolveCallable());
    if (!otherFunc) {
      return WalkResult::advance();
    }
    for (Value otherValue : otherCall->getResults()) {
      auto otherResult = llvm::cast<OpResult>(otherValue);
      auto otherResName = otherFunc.getResNameAttr(otherResult.getResultNumber());
      if (!otherResName || otherResName->getValue() != resName->getValue()) {
        continue;
      }
      if (otherResult == callResult) {
        foundThisCall = true;
        continue;
      }
      foundDuplicate = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return foundThisCall && !foundDuplicate;
}

void SourceRef::print(raw_ostream &os) const {
  if (isConstantFelt()) {
    os << "<felt.const: " << *getConstantFeltValue() << '>';
  } else if (isConstantIndex()) {
    os << "<index: " << *getConstantIndexValue() << '>';
  } else if (isTemplateConstant()) {
    auto constRead = getDefiningOp<ConstReadOp>();
    ensure(succeeded(constRead), "template constant should be backed by a const.read op");
    auto structDefOp = (*constRead)->getParentOfType<StructDefOp>();
    ensure(structDefOp, "struct template should have a struct parent");
    os << '@' << structDefOp.getName() << "<[@" << constRead->getConstName() << "]>";
  } else {
    if (isCreateStructOp()) {
      os << "%self";
    } else if (isBlockArgument()) {
      auto blockArg = *getBlockArgument();
      auto funcOp = llvm::dyn_cast<FuncDefOp>(blockArg.getOwner()->getParentOp());
      auto argName = funcOp ? funcOp.getArgNameAttr(blockArg.getArgNumber()) : nullptr;
      if (argName) {
        os << argName->getValue();
      } else {
        os << "%arg" << *getInputNum();
      }
    } else if (isNonDetOp()) {
      os << '<' << *getNonDetOp() << '>';
    } else if (isCallResult()) {
      auto callOp = *getCallOp();
      auto callResult = llvm::cast<OpResult>(value);
      auto callee = resolveCallable<FuncDefOp>(callOp);
      if (succeeded(callee)) {
        auto calleeFunc = llvm::dyn_cast_if_present<FuncDefOp>((*callee).get());
        if (shouldPrintNamedCallResult(callOp, callResult, calleeFunc)) {
          auto resName = *calleeFunc.getResNameAttr(callResult.getResultNumber());
          os << resName.getValue();
        } else {
          printCallResultFallback(os, callOp, value);
        }
      } else {
        printCallResultFallback(os, callOp, value);
      }
    } else {
      ensure(isRooted(), "unhandled print case");
      OpPrintingFlags flags;
      value.printAsOperand(os, flags);
    }

    auto res = printSourceStylePath(os, getPath());
    ensure(succeeded(res), "unhandled path print case");
  }
}

bool SourceRef::operator==(const SourceRef &rhs) const {
  // This way two felt constants can be equal even if the declared in separate ops.
  if (isConstantInt() && rhs.isConstantInt()) {
    DynamicAPInt lhsVal = *getConstantValue(), rhsVal = *rhs.getConstantValue();
    return getType() == rhs.getType() && lhsVal == rhsVal;
  }
  return constant == rhs.constant && value == rhs.value && llvm::equal(getPath(), rhs.getPath());
}

// required for EquivalenceClasses usage
std::strong_ordering SourceRef::operator<=>(const SourceRef &rhs) const {
  auto lhsCategory = getSortCategory();
  auto rhsCategory = rhs.getSortCategory();
  if (auto cmp = lhsCategory <=> rhsCategory; cmp != std::strong_ordering::equal) {
    return cmp;
  }
  return compareWithinCategory(rhs, lhsCategory);
}

size_t SourceRef::Hash::operator()(const SourceRef &val) const {
  if (val.isConstantInt()) {
    return llvm::hash_combine(val.getType(), *val.getConstantValue());
  } else if (val.isTemplateConstant()) {
    return llvm::hash_value(val.getAsOpaquePointer());
  } else {
    ensure(
        val.isBlockArgument() || val.isCreateStructOp() || val.isNonDetOp() || val.isRooted(),
        "unhandled SourceRef hash case"
    );

    size_t hash = llvm::hash_value(val.getAsOpaquePointer());
    for (const auto &f : val.getPath()) {
      hash = llvm::hash_combine(hash, f.getHash());
    }
    return hash;
  }
}

raw_ostream &operator<<(raw_ostream &os, const SourceRef &rhs) {
  rhs.print(os);
  return os;
}

/* SourceRefSet */

SourceRefSet &SourceRefSet::join(const SourceRefSet &rhs) {
  insert(rhs.begin(), rhs.end());
  return *this;
}

raw_ostream &operator<<(raw_ostream &os, const SourceRefSet &rhs) {
  os << "{ ";
  std::vector<SourceRef> sortedRefs(rhs.begin(), rhs.end());
  std::sort(sortedRefs.begin(), sortedRefs.end());
  for (auto it = sortedRefs.begin(); it != sortedRefs.end();) {
    os << *it;
    it++;
    if (it != sortedRefs.end()) {
      os << ", ";
    } else {
      os << ' ';
    }
  }
  os << '}';
  return os;
}

} // namespace llzk
