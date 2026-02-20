//===-- SymbolHelper.cpp - LLZK Symbol Helpers ------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementations for symbol helper functions.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "llzk-symbol-helpers"

using namespace mlir;

namespace llzk {

using namespace array;
using namespace component;
using namespace function;
using namespace global;
using namespace polymorphic;

namespace {

// NOTE: These may be used in SymbolRefAttr instances returned from these functions but there is no
// restriction that the same value cannot be used as a symbol name in user code so these should not
// be used in such a way that relies on that assumption. That's why they are (currently) defined in
// this anonymous namespace rather than within the header file.
constexpr char POSITION_IS_ROOT_INDICATOR[] = "<<symbol lookup root>>";
constexpr char UNNAMED_SYMBOL_INDICATOR[] = "<<unnamed symbol>>";

enum RootSelector : std::uint8_t { CLOSEST, FURTHEST };

class RootPathBuilder {
  RootSelector _whichRoot;
  Operation *_origin;
  ModuleOp *_foundRoot;

public:
  RootPathBuilder(RootSelector whichRoot, Operation *origin, ModuleOp *foundRoot)
      : _whichRoot(whichRoot), _origin(origin), _foundRoot(foundRoot) {}

  /// Traverse ModuleOp ancestors of `from` and add their names to `path` until the (closest or
  /// furthest, based on RootSelector argument) ModuleOp with the `LANG_ATTR_NAME` attribute is
  /// reached. If a ModuleOp without a name is reached or a ModuleOp with the `LANG_ATTR_NAME`
  /// attribute is never found, produce an error (referencing the `origin` Operation). The name
  /// of the root module itself is not added to the path.
  ///
  /// Returns the module containing the LANG_ATTR_NAME attribute.
  FailureOr<ModuleOp> collectPathToRoot(Operation *from, std::vector<FlatSymbolRefAttr> &path) {
    Operation *check = from;
    ModuleOp currRoot = nullptr;
    do {
      if (ModuleOp m = llvm::dyn_cast_if_present<ModuleOp>(check)) {
        // We need this attribute restriction because some stages of parsing have
        //  an extra module wrapping the top-level module from the input file.
        // This module, even if it has a name, does not contribute to path names.
        if (m->hasAttr(LANG_ATTR_NAME)) {
          if (_whichRoot == RootSelector::CLOSEST) {
            return m;
          }
          currRoot = m;
        }
        if (StringAttr modName = m.getSymNameAttr()) {
          path.push_back(FlatSymbolRefAttr::get(modName));
        } else if (!currRoot) {
          return _origin->emitOpError()
              .append(
                  "has ancestor '", ModuleOp::getOperationName(), "' without \"", LANG_ATTR_NAME,
                  "\" attribute or a name"
              )
              .attachNote(m.getLoc())
              .append("unnamed '", ModuleOp::getOperationName(), "' here");
        }
      }
    } while ((check = check->getParentOp()));

    if (_whichRoot == RootSelector::FURTHEST && currRoot) {
      return currRoot;
    }

    return _origin->emitOpError().append(
        "has no ancestor '", ModuleOp::getOperationName(), "' with \"", LANG_ATTR_NAME,
        "\" attribute"
    );
  }

  /// Appends to the `path` argument via `collectPathToRoot()` starting from `position` and then
  /// convert that path into a SymbolRefAttr.
  FailureOr<SymbolRefAttr> buildPathFromRootToAnyOp(
      Operation *position, std::vector<FlatSymbolRefAttr> &&path

  ) {
    // Collect the rest of the path to the root module
    FailureOr<ModuleOp> rootMod = collectPathToRoot(position, path);
    if (failed(rootMod)) {
      return failure();
    }
    if (_foundRoot) {
      *_foundRoot = rootMod.value();
    }
    // Special case for empty path (because asSymbolRefAttr() cannot handle it).
    if (path.empty()) {
      // ASSERT: This can only occur when the given `position` is the discovered root ModuleOp
      // itself.
      assert(position == rootMod.value().getOperation() && "empty path only at root itself");
      return getFlatSymbolRefAttr(_origin->getContext(), POSITION_IS_ROOT_INDICATOR);
    }
    //  Reverse the vector and convert it to a SymbolRefAttr
    std::vector<FlatSymbolRefAttr> reversedVec(path.rbegin(), path.rend());
    return asSymbolRefAttr(reversedVec);
  }

  /// Appends the `path` via `collectPathToRoot()` starting from the given `StructDefOp` and then
  /// convert that path into a SymbolRefAttr.
  FailureOr<SymbolRefAttr>
  buildPathFromRootToStruct(StructDefOp to, std::vector<FlatSymbolRefAttr> &&path) {
    // Add the name of the struct (its name is not optional) and then delegate to helper
    path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));
    return buildPathFromRootToAnyOp(to, std::move(path));
  }

  FailureOr<SymbolRefAttr> getPathFromRootToStruct(StructDefOp to) {
    std::vector<FlatSymbolRefAttr> path;
    return buildPathFromRootToStruct(to, std::move(path));
  }

  FailureOr<SymbolRefAttr> getPathFromRootToMember(MemberDefOp to) {
    std::vector<FlatSymbolRefAttr> path;
    // Add the name of the member (its name is not optional)
    path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));
    // Delegate to the parent handler (must be StructDefOp per ODS)
    return buildPathFromRootToStruct(to.getParentOp<StructDefOp>(), std::move(path));
  }

  FailureOr<SymbolRefAttr> getPathFromRootToFunc(FuncDefOp to) {
    std::vector<FlatSymbolRefAttr> path;
    // Add the name of the function (its name is not optional)
    path.push_back(FlatSymbolRefAttr::get(to.getSymNameAttr()));

    // Delegate based on the type of the parent op
    Operation *current = to.getOperation();
    Operation *parent = current->getParentOp();
    if (StructDefOp parentStruct = llvm::dyn_cast_if_present<StructDefOp>(parent)) {
      return buildPathFromRootToStruct(parentStruct, std::move(path));
    } else if (ModuleOp parentMod = llvm::dyn_cast_if_present<ModuleOp>(parent)) {
      return buildPathFromRootToAnyOp(parentMod, std::move(path));
    } else {
      // This is an error in the compiler itself. In current implementation,
      //  FuncDefOp must have either StructDefOp or ModuleOp as its parent.
      return current->emitError().append("orphaned '", FuncDefOp::getOperationName(), '\'');
    }
  }

  FailureOr<SymbolRefAttr> getPathFromRootToAnySymbol(SymbolOpInterface to) {
    return TypeSwitch<Operation *, FailureOr<SymbolRefAttr>>(to.getOperation())
        // This more general function must check for the specific cases first.
        .Case<FuncDefOp>([this](FuncDefOp toOp) { return getPathFromRootToFunc(toOp); })
        .Case<MemberDefOp>([this](MemberDefOp toOp) { return getPathFromRootToMember(toOp); })
        .Case<StructDefOp>([this](StructDefOp toOp) { return getPathFromRootToStruct(toOp); })

        // If it's a module, immediately delegate to `buildPathFromRootToAnyOp()` since
        // it will already add the module name to the path.
        .Case<ModuleOp>([this](ModuleOp toOp) {
      std::vector<FlatSymbolRefAttr> path;
      return buildPathFromRootToAnyOp(toOp, std::move(path));
    })

        // For any other symbol, append the name of the symbol and then delegate to
        // `buildPathFromRootToAnyOp()`.
        .Default([this, &to](Operation *) {
      std::vector<FlatSymbolRefAttr> path;
      if (StringAttr name = llzk::getSymbolName(to)) {
        path.push_back(FlatSymbolRefAttr::get(name));
      } else {
        // This can only happen if the symbol is optional. Add a placeholder name.
        assert(to.isOptionalSymbol());
        path.push_back(FlatSymbolRefAttr::get(to.getContext(), UNNAMED_SYMBOL_INDICATOR));
      }
      return buildPathFromRootToAnyOp(to, std::move(path));
    });
  }
};

} // namespace

llvm::SmallVector<StringRef> getNames(SymbolRefAttr ref) {
  llvm::SmallVector<StringRef> names;
  names.push_back(ref.getRootReference().getValue());
  for (const FlatSymbolRefAttr &r : ref.getNestedReferences()) {
    names.push_back(r.getValue());
  }
  return names;
}

llvm::SmallVector<FlatSymbolRefAttr> getPieces(SymbolRefAttr ref) {
  llvm::SmallVector<FlatSymbolRefAttr> pieces;
  pieces.push_back(FlatSymbolRefAttr::get(ref.getRootReference()));
  for (const FlatSymbolRefAttr &r : ref.getNestedReferences()) {
    pieces.push_back(r);
  }
  return pieces;
}

namespace {

SymbolRefAttr changeLeafImpl(
    StringAttr origRoot, ArrayRef<FlatSymbolRefAttr> origTail, FlatSymbolRefAttr newLeaf,
    size_t drop = 1
) {
  llvm::SmallVector<FlatSymbolRefAttr> newTail;
  newTail.append(origTail.begin(), origTail.drop_back(drop).end());
  newTail.push_back(newLeaf);
  return SymbolRefAttr::get(origRoot, newTail);
}

} // namespace

SymbolRefAttr replaceLeaf(SymbolRefAttr orig, FlatSymbolRefAttr newLeaf) {
  ArrayRef<FlatSymbolRefAttr> origTail = orig.getNestedReferences();
  if (origTail.empty()) {
    // If there is no tail, the root is the leaf so replace the whole thing
    return newLeaf;
  } else {
    return changeLeafImpl(orig.getRootReference(), origTail, newLeaf);
  }
}

SymbolRefAttr appendLeaf(SymbolRefAttr orig, FlatSymbolRefAttr newLeaf) {
  return changeLeafImpl(orig.getRootReference(), orig.getNestedReferences(), newLeaf, 0);
}

SymbolRefAttr appendLeafName(SymbolRefAttr orig, const Twine &newLeafSuffix) {
  ArrayRef<FlatSymbolRefAttr> origTail = orig.getNestedReferences();
  if (origTail.empty()) {
    // If there is no tail, the root is the leaf so append on the root instead
    return getFlatSymbolRefAttr(
        orig.getContext(), orig.getRootReference().getValue() + newLeafSuffix
    );
  } else {
    return changeLeafImpl(
        orig.getRootReference(), origTail,
        getFlatSymbolRefAttr(orig.getContext(), origTail.back().getValue() + newLeafSuffix)
    );
  }
}

FailureOr<ModuleOp> getRootModule(Operation *from) {
  std::vector<FlatSymbolRefAttr> path;
  return RootPathBuilder(RootSelector::CLOSEST, from, nullptr).collectPathToRoot(from, path);
}

FailureOr<SymbolRefAttr> getPathFromRoot(SymbolOpInterface to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::CLOSEST, to, foundRoot).getPathFromRootToAnySymbol(to);
}

FailureOr<SymbolRefAttr> getPathFromRoot(StructDefOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::CLOSEST, to, foundRoot).getPathFromRootToStruct(to);
}

FailureOr<SymbolRefAttr> getPathFromRoot(MemberDefOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::CLOSEST, to, foundRoot).getPathFromRootToMember(to);
}

FailureOr<SymbolRefAttr> getPathFromRoot(FuncDefOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::CLOSEST, to, foundRoot).getPathFromRootToFunc(to);
}

FailureOr<ModuleOp> getTopRootModule(Operation *from) {
  std::vector<FlatSymbolRefAttr> path;
  return RootPathBuilder(RootSelector::FURTHEST, from, nullptr).collectPathToRoot(from, path);
}

FailureOr<SymbolRefAttr> getPathFromTopRoot(SymbolOpInterface to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::FURTHEST, to, foundRoot).getPathFromRootToAnySymbol(to);
}

FailureOr<SymbolRefAttr> getPathFromTopRoot(StructDefOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::FURTHEST, to, foundRoot).getPathFromRootToStruct(to);
}

FailureOr<SymbolRefAttr> getPathFromTopRoot(MemberDefOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::FURTHEST, to, foundRoot).getPathFromRootToMember(to);
}

FailureOr<SymbolRefAttr> getPathFromTopRoot(FuncDefOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::FURTHEST, to, foundRoot).getPathFromRootToFunc(to);
}

FailureOr<StructType> getMainInstanceType(Operation *lookupFrom) {
  FailureOr<ModuleOp> rootOpt = getRootModule(lookupFrom);
  if (failed(rootOpt)) {
    return failure();
  }
  ModuleOp root = rootOpt.value();
  if (Attribute a = root->getAttr(MAIN_ATTR_NAME)) {
    // If the attribute is present, it must be a TypeAttr of concrete StructType.
    if (TypeAttr ta = llvm::dyn_cast<TypeAttr>(a)) {
      if (StructType st = llvm::dyn_cast<StructType>(ta.getValue())) {
        if (isConcreteType(st)) {
          return success(st);
        }
      }
    }
    return rootOpt->emitError().append(
        '"', MAIN_ATTR_NAME, "\" on top-level module must be a concrete '", StructType::name,
        "' attribute. Found: ", a
    );
  }
  // The attribute is optional so it's okay if not present.
  return success(nullptr);
}

FailureOr<SymbolLookupResult<StructDefOp>>
getMainInstanceDef(SymbolTableCollection &symbolTable, Operation *lookupFrom) {
  FailureOr<StructType> mainStructTypeOpt = getMainInstanceType(lookupFrom);
  if (failed(mainStructTypeOpt)) {
    return failure();
  }
  if (StructType st = mainStructTypeOpt.value()) {
    return st.getDefinition(symbolTable, lookupFrom);
  } else {
    return success(nullptr);
  }
}

LogicalResult verifyParamOfType(
    SymbolTableCollection &tables, SymbolRefAttr param, Type parameterizedType, Operation *origin
) {
  // Most often, StructType and ArrayType SymbolRefAttr parameters will be defined as parameters of
  // the StructDefOp that the current Operation is nested within. These are always flat references
  // (i.e., contain no nested references).
  if (param.getNestedReferences().empty()) {
    FailureOr<StructDefOp> getParentRes = getParentOfType<StructDefOp>(origin);
    if (succeeded(getParentRes)) {
      if (getParentRes->hasParamNamed(param.getRootReference())) {
        return success();
      }
    }
  }
  // Otherwise, see if the symbol can be found via lookup from the `origin` Operation.
  auto lookupRes = lookupTopLevelSymbol(tables, param, origin);
  if (failed(lookupRes)) {
    return failure(); // lookupTopLevelSymbol() already emits a sufficient error message
  }
  Operation *foundOp = lookupRes->get();
  if (!llvm::isa<GlobalDefOp>(foundOp)) {
    return origin->emitError() << "ref \"" << param << "\" in type " << parameterizedType
                               << " refers to a '" << foundOp->getName()
                               << "' which is not allowed";
  }
  return success();
}

LogicalResult verifyParamsOfType(
    SymbolTableCollection &tables, ArrayRef<Attribute> tyParams, Type parameterizedType,
    Operation *origin
) {
  // Rather than immediately returning on failure, we check all params and aggregate to provide as
  // many errors are possible in a single verifier run.
  LogicalResult paramCheckResult = success();
  LLVM_DEBUG({
    llvm::dbgs() << "[verifyParamOfType] parameterizedType = " << parameterizedType << '\n';
  });
  for (Attribute attr : tyParams) {
    LLVM_DEBUG({ llvm::dbgs() << "[verifyParamOfType]   checking attribute " << attr << '\n'; });
    assertValidAttrForParamOfType(attr);
    if (SymbolRefAttr symRefParam = llvm::dyn_cast<SymbolRefAttr>(attr)) {
      if (failed(verifyParamOfType(tables, symRefParam, parameterizedType, origin))) {
        LLVM_DEBUG({
          llvm::dbgs() << "[verifyParamOfType]     failed to verify symbol attribute\n";
        });
        paramCheckResult = failure();
      }
    } else if (TypeAttr typeParam = llvm::dyn_cast<TypeAttr>(attr)) {
      if (failed(verifyTypeResolution(tables, origin, typeParam.getValue()))) {
        LLVM_DEBUG({
          llvm::dbgs() << "[verifyParamOfType]     failed to verify type attribute\n";
        });
        paramCheckResult = failure();
      }
    }
    LLVM_DEBUG({ llvm::dbgs() << "[verifyParamOfType]     verified attribute\n"; });
    // IntegerAttr and AffineMapAttr cannot contain symbol references
  }
  return paramCheckResult;
}

FailureOr<StructDefOp>
verifyStructTypeResolution(SymbolTableCollection &tables, StructType ty, Operation *origin) {
  auto res = ty.getDefinition(tables, origin);
  if (failed(res)) {
    return failure();
  }
  StructDefOp defForType = res.value().get();
  if (!structTypesUnify(ty, defForType.getType({}), res->getNamespace())) {
    return origin->emitError()
        .append(
            "Cannot unify parameters of type ", ty, " with parameters of '",
            StructDefOp::getOperationName(), "' \"", defForType.getHeaderString(), '"'
        )
        .attachNote(defForType.getLoc())
        .append("type parameters must unify with parameters defined here");
  }
  // If there are any SymbolRefAttr parameters on the StructType, ensure those refs are valid.
  if (ArrayAttr tyParams = ty.getParams()) {
    if (failed(verifyParamsOfType(tables, tyParams.getValue(), ty, origin))) {
      return failure(); // verifyParamsOfType() already emits a sufficient error message
    }
  }
  return defForType;
}

LogicalResult verifyTypeResolution(SymbolTableCollection &tables, Operation *origin, Type ty) {
  if (StructType sTy = llvm::dyn_cast<StructType>(ty)) {
    return verifyStructTypeResolution(tables, sTy, origin);
  } else if (ArrayType aTy = llvm::dyn_cast<ArrayType>(ty)) {
    if (failed(verifyParamsOfType(tables, aTy.getDimensionSizes(), aTy, origin))) {
      return failure();
    }
    return verifyTypeResolution(tables, origin, aTy.getElementType());
  } else if (TypeVarType vTy = llvm::dyn_cast<TypeVarType>(ty)) {
    return verifyParamOfType(tables, vTy.getNameRef(), vTy, origin);
  } else {
    return success();
  }
}

} // namespace llzk
