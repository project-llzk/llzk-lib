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

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace felt;
using namespace function;
using namespace polymorphic;
using namespace string;

/* SourceRefIndex */

void SourceRefIndex::print(raw_ostream &os) const {
  if (isMember()) {
    os << '@' << getMember().getName();
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

bool SourceRefIndex::operator<(const SourceRefIndex &rhs) const {
  if (isMember() && rhs.isMember()) {
    return NamedOpLocationLess<MemberDefOp> {}(getMember(), rhs.getMember());
  }
  if (isIndex() && rhs.isIndex()) {
    return getIndex() < rhs.getIndex();
  }
  if (isIndexRange() && rhs.isIndexRange()) {
    auto [ll, lu] = getIndexRange();
    auto [rl, ru] = rhs.getIndexRange();
    return ll < rl || (ll == rl && lu < ru);
  }

  if (isMember()) {
    return true;
  }
  if (isIndex() && !rhs.isMember()) {
    return true;
  }

  return false;
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
  } else {
    return OpHash<component::MemberDefOp> {}(c.getMember());
  }
}

/* SourceRef */

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

  return std::move(sDef.value());
}

std::vector<SourceRef>
SourceRef::getAllSourceRefs(SymbolTableCollection &tables, ModuleOp mod, SourceRef root) {
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
  if (isConstantFelt()) {
    return std::get<FeltConstantOp>(*constantVal).getType();
  } else if (isConstantIndex()) {
    return std::get<arith::ConstantIndexOp>(*constantVal).getType();
  } else if (isTemplateConstant()) {
    return std::get<ConstReadOp>(*constantVal).getType();
  } else {
    int array_derefs = 0;
    int idx = memberRefs.size() - 1;
    while (idx >= 0 && memberRefs[idx].isIndex()) {
      array_derefs++;
      idx--;
    }

    Type currTy = nullptr;
    if (idx >= 0) {
      currTy = memberRefs[idx].getMember().getType();
    } else {
      currTy = isBlockArgument() ? getBlockArgument().getType() : getCreateStructOp().getType();
    }

    while (array_derefs > 0) {
      currTy = dyn_cast<ArrayType>(currTy).getElementType();
      array_derefs--;
    }
    return currTy;
  }
}

bool SourceRef::isValidPrefix(const SourceRef &prefix) const {
  if (isConstant()) {
    return false;
  }

  if (root != prefix.root || memberRefs.size() < prefix.memberRefs.size()) {
    return false;
  }
  for (size_t i = 0; i < prefix.memberRefs.size(); i++) {
    if (memberRefs[i] != prefix.memberRefs[i]) {
      return false;
    }
  }
  return true;
}

FailureOr<std::vector<SourceRefIndex>> SourceRef::getSuffix(const SourceRef &prefix) const {
  if (!isValidPrefix(prefix)) {
    return failure();
  }
  std::vector<SourceRefIndex> suffix;
  for (size_t i = prefix.memberRefs.size(); i < memberRefs.size(); i++) {
    suffix.push_back(memberRefs[i]);
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

  auto newSignalUsage = other;
  newSignalUsage.memberRefs.insert(newSignalUsage.memberRefs.end(), suffix->begin(), suffix->end());
  return newSignalUsage;
}

std::vector<SourceRef>
getAllChildren(SymbolTableCollection &tables, ModuleOp /*mod*/, ArrayType arrayTy, SourceRef root) {
  std::vector<SourceRef> res;
  // Recurse into arrays by iterating over their elements
  for (int64_t i = 0; i < arrayTy.getDimSize(0); i++) {
    SourceRef childRef = root.createChild(SourceRefIndex(i));
    res.push_back(childRef);
  }

  return res;
}

std::vector<SourceRef> getAllChildren(
    SymbolTableCollection &tables, ModuleOp mod, SymbolLookupResult<StructDefOp> structDefRes,
    SourceRef root
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
    SourceRef childRef = root.createChild(SourceRefIndex(memberLookup.value()));
    // Make a reference to the current member, regardless of if it is a composite
    // type or not.
    res.push_back(childRef);
  }
  return res;
}

std::vector<SourceRef>
SourceRef::getAllChildren(SymbolTableCollection &tables, ModuleOp mod) const {
  auto ty = getType();
  if (auto structTy = dyn_cast<StructType>(ty)) {
    return llzk::getAllChildren(tables, mod, getStructDef(tables, mod, structTy), *this);
  } else if (auto arrayType = dyn_cast<ArrayType>(ty)) {
    return llzk::getAllChildren(tables, mod, arrayType, *this);
  }
  // Scalar type, no children
  return {};
}

void SourceRef::print(raw_ostream &os) const {
  if (isConstantFelt()) {
    os << "<felt.const: " << getConstantFeltValue() << '>';
  } else if (isConstantIndex()) {
    os << "<index: " << getConstantIndexValue() << '>';
  } else if (isTemplateConstant()) {
    auto constRead = std::get<ConstReadOp>(*constantVal);
    auto structDefOp = constRead->getParentOfType<StructDefOp>();
    ensure(structDefOp, "struct template should have a struct parent");
    os << '@' << structDefOp.getName() << "<[@" << constRead.getConstName() << "]>";
  } else {
    if (isCreateStructOp()) {
      os << "%self";
    } else {
      ensure(isBlockArgument(), "unhandled print case");
      os << "%arg" << getInputNum();
    }

    for (auto f : memberRefs) {
      os << "[" << f << "]";
    }
  }
}

bool SourceRef::operator==(const SourceRef &rhs) const {
  // This way two felt constants can be equal even if the declared in separate ops.
  if (isConstantInt() && rhs.isConstantInt()) {
    DynamicAPInt lhsVal = getConstantValue(), rhsVal = rhs.getConstantValue();
    return getType() == rhs.getType() && lhsVal == rhsVal;
  }
  return (root == rhs.root) && (memberRefs == rhs.memberRefs) && (constantVal == rhs.constantVal);
}

// required for EquivalenceClasses usage
bool SourceRef::operator<(const SourceRef &rhs) const {
  if (isConstantFelt() && !rhs.isConstantFelt()) {
    // Put all constants at the end
    return false;
  } else if (!isConstantFelt() && rhs.isConstantFelt()) {
    return true;
  } else if (isConstantFelt() && rhs.isConstantFelt()) {
    DynamicAPInt lhsInt = getConstantFeltValue(), rhsInt = rhs.getConstantFeltValue();
    return lhsInt < rhsInt;
  }

  if (isConstantIndex() && !rhs.isConstantIndex()) {
    // Put all constant indices next at the end
    return false;
  } else if (!isConstantIndex() && rhs.isConstantIndex()) {
    return true;
  } else if (isConstantIndex() && rhs.isConstantIndex()) {
    DynamicAPInt lhsVal = getConstantIndexValue(), rhsVal = rhs.getConstantIndexValue();
    return lhsVal < rhsVal;
  }

  if (isTemplateConstant() && !rhs.isTemplateConstant()) {
    // Put all template constants next at the end
    return false;
  } else if (!isTemplateConstant() && rhs.isTemplateConstant()) {
    return true;
  } else if (isTemplateConstant() && rhs.isTemplateConstant()) {
    StringRef lhsName = std::get<ConstReadOp>(*constantVal).getConstName();
    StringRef rhsName = std::get<ConstReadOp>(*rhs.constantVal).getConstName();
    return lhsName.compare(rhsName) < 0;
  }

  // Sort out the block argument vs struct.new cases
  if (isBlockArgument() && rhs.isCreateStructOp()) {
    return true;
  } else if (isCreateStructOp() && rhs.isBlockArgument()) {
    return false;
  } else if (isBlockArgument() && rhs.isBlockArgument()) {
    if (getInputNum() < rhs.getInputNum()) {
      return true;
    } else if (getInputNum() > rhs.getInputNum()) {
      return false;
    }
  } else if (isCreateStructOp() && rhs.isCreateStructOp()) {
    CreateStructOp lhsOp = getCreateStructOp(), rhsOp = rhs.getCreateStructOp();
    if (lhsOp < rhsOp) {
      return true;
    } else if (lhsOp > rhsOp) {
      return false;
    }
  } else {
    llvm_unreachable("unhandled operator< case");
  }

  for (size_t i = 0; i < memberRefs.size() && i < rhs.memberRefs.size(); i++) {
    if (memberRefs[i] < rhs.memberRefs[i]) {
      return true;
    } else if (memberRefs[i] > rhs.memberRefs[i]) {
      return false;
    }
  }
  return memberRefs.size() < rhs.memberRefs.size();
}

size_t SourceRef::Hash::operator()(const SourceRef &val) const {
  if (val.isConstantInt()) {
    return llvm::hash_value(val.getConstantValue());
  } else if (val.isTemplateConstant()) {
    return OpHash<ConstReadOp> {}(std::get<ConstReadOp>(*val.constantVal));
  } else {
    ensure(val.isBlockArgument() || val.isCreateStructOp(), "unhandled SourceRef hash case");

    size_t hash = val.isBlockArgument() ? std::hash<unsigned> {}(val.getInputNum())
                                        : OpHash<CreateStructOp> {}(val.getCreateStructOp());
    for (auto f : val.memberRefs) {
      hash ^= f.getHash();
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
