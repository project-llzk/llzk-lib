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

#include "llzk/Util/SymbolHelper.h"

#include "llzk/Dialect/Array/IR/Ops.h"
#include "llzk/Dialect/Felt/IR/Types.h"
#include "llzk/Dialect/Function/IR/Ops.h"
#include "llzk/Dialect/Global/IR/Ops.h"
#include "llzk/Dialect/Polymorphic/IR/Types.h"
#include "llzk/Dialect/Verif/IR/Ops.h"
#include "llzk/Util/SymbolLookup.h"
#include "llzk/Util/SymbolTableLLZK.h"
#include "llzk/Util/TypeHelper.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/TypeSwitch.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/ErrorHandling.h>

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
      } else if (TemplateOp t = llvm::dyn_cast_if_present<TemplateOp>(check)) {
        StringAttr name = t.getSymNameAttr();
        assert(name && "per ODS");
        path.push_back(FlatSymbolRefAttr::get(name));
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
  FailureOr<SymbolRefAttr>
  buildPathFromRootToAnyOp(Operation *position, std::vector<FlatSymbolRefAttr> &&path) {
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

  /// For cases where the current op name will already be added by `buildPathFromRootToAnyOp()`.
  FailureOr<SymbolRefAttr> getPathFromRootToAnyOp(Operation *op) {
    std::vector<FlatSymbolRefAttr> path;
    return buildPathFromRootToAnyOp(op, std::move(path));
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
    } else if (TemplateOp parentTemplate = llvm::dyn_cast_if_present<TemplateOp>(parent)) {
      return buildPathFromRootToAnyOp(parentTemplate, std::move(path));
    } else {
      // This is an error in the compiler itself. In current implementation,
      //  FuncDefOp must have module, struct, or template as its parent.
      return current->emitError().append("orphaned '", FuncDefOp::getOperationName(), '\'');
    }
  }

  FailureOr<SymbolRefAttr> getPathFromRootToAnySymbol(SymbolOpInterface to) {
    // clang-format off
    return TypeSwitch<Operation *, FailureOr<SymbolRefAttr>>(to.getOperation())
      // This more general function must check for the specific cases first.
      .Case<FuncDefOp>([this](auto toOp) { return getPathFromRootToFunc(toOp); })
      .Case<MemberDefOp>([this](auto toOp) { return getPathFromRootToMember(toOp); })
      .Case<StructDefOp>([this](auto toOp) { return getPathFromRootToStruct(toOp); })
      .Case<TemplateOp>([this](auto toOp) { return getPathFromRootToAnyOp(toOp); })
      .Case<ModuleOp>([this](auto toOp) { return getPathFromRootToAnyOp(toOp); })

      // For any other symbol, append the name of the symbol and then delegate to
      // `buildPathFromRootToAnyOp()`.
      .Default([this, &to](auto) {
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
    // clang-format on
  }
};

LogicalResult verifyTemplateSymbolType(
    TemplateSymbolBindingOpInterface binding, SymbolRefAttr param, Type parameterizedType,
    Operation *origin, std::optional<Type> requiredParamType,
    std::optional<Location> requiredParamLoc
) {
  if (requiredParamType) {
    std::optional<Type> actualType = binding.getTypeOpt();
    // A direct array-dimension symbol must establish its index kind at the use site. Other
    // template arguments may remain unrestricted until their enclosing template is specialized.
    bool missingArrayDimensionType = !actualType && llvm::isa<array::ArrayType>(parameterizedType);
    if (missingArrayDimensionType ||
        !isTemplateParamTypeCompatible(actualType, *requiredParamType)) {
      if (!actualType) {
        auto diag = origin->emitError().append(
            "ref \"", param, "\" in type ", parameterizedType, " refers to a '", binding->getName(),
            "' that must have type ", *requiredParamType
        );
        diag.attachNote(binding->getLoc()).append("referenced binding declared here");
        if (requiredParamLoc) {
          diag.attachNote(*requiredParamLoc).append("required parameter declared here");
        }
        return diag;
      }
      auto diag = origin->emitError().append(
          "ref \"", param, "\" in type ", parameterizedType, " refers to a '", binding->getName(),
          "' with type ", *actualType, " but expected ", *requiredParamType
      );
      diag.attachNote(binding->getLoc()).append("referenced binding declared here");
      if (requiredParamLoc) {
        diag.attachNote(*requiredParamLoc).append("required parameter declared here");
      }
      return diag;
    }
  }
  return success();
}

FailureOr<bool> resolvedTemplateParamValuesUnify(
    SymbolTableCollection &tables, Operation *origin, Attribute explicitValue,
    Attribute inferredValue, std::optional<Type> requiredParamType
);

/// Verify that repeated felt candidates satisfy the declared restriction and mutually agree after
/// resolving contextual symbol evidence. When an explicit value is present, require it to agree
/// with every candidate. This preserves generic ambiguity for non-felt parameters.
LogicalResult verifyRepeatedFeltCandidates(
    Operation *origin, TemplateParamOp paramOp, ArrayRef<Attribute> inferredCandidates,
    StringRef signatureDescription, Attribute explicitValue = nullptr
) {
  for (Attribute candidate : inferredCandidates) {
    if (failed(verifyTemplateParamValueCompatibility(origin, candidate, paramOp))) {
      return failure();
    }
  }

  SymbolTableCollection tables;
  if (explicitValue) {
    for (Attribute inferredCandidate : inferredCandidates) {
      FailureOr<bool> resolved = resolvedTemplateParamValuesUnify(
          tables, origin, explicitValue, inferredCandidate, paramOp.getTypeOpt()
      );
      if (failed(resolved)) {
        return failure();
      }
      if (!*resolved) {
        return origin->emitOpError().append(
            "template instantiation value '", explicitValue, "' for parameter \"@",
            paramOp.getName(), "\" conflicts with value '", inferredCandidate, "' inferred from ",
            signatureDescription, " type signature"
        );
      }
    }
  }

  // Compare every pair because compatibility is not transitive: one symbolic felt can be
  // compatible with two concrete values that conflict with each other.
  for (auto [i, lhsCandidate] : llvm::enumerate(inferredCandidates)) {
    for (Attribute rhsCandidate : inferredCandidates.drop_front(i + 1)) {
      FailureOr<bool> resolved = resolvedTemplateParamValuesUnify(
          tables, origin, lhsCandidate, rhsCandidate, paramOp.getTypeOpt()
      );
      if (failed(resolved)) {
        return failure();
      }
      if (!*resolved) {
        return origin->emitOpError().append(
            "cannot infer template instantiation value for parameter \"@", paramOp.getName(),
            "\" from ", signatureDescription, " type signature"
        );
      }
    }
  }
  return success();
}

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

FailureOr<SymbolRefAttr> getPathFromRoot(TemplateOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::CLOSEST, to, foundRoot).getPathFromRootToAnyOp(to);
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

FailureOr<SymbolRefAttr> getPathFromTopRoot(TemplateOp &to, ModuleOp *foundRoot) {
  return RootPathBuilder(RootSelector::FURTHEST, to, foundRoot).getPathFromRootToAnyOp(to);
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
    return getTypeFromLlzkMainAttr(root, a);
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

FailureOr<TemplateOp> getConstResolutionTemplate(SymbolTableCollection &tables, Operation *origin) {
  if (auto contract = origin->getParentOfType<verif::ContractOp>()) {
    FailureOr<SymbolLookupResultUntyped> targetRes =
        lookupTopLevelSymbol(tables, contract.getTargetAttr(), origin);
    if (failed(targetRes)) {
      return failure(); // lookupTopLevelSymbol() already emits a sufficient error message
    }

    if (TemplateOp targetTemplate = targetRes->get()->getParentOfType<TemplateOp>()) {
      return targetTemplate;
    }
  }

  return getParentOfType<TemplateOp>(origin);
}

LogicalResult verifyTemplateParamValueCompatibility(
    Operation *origin, Attribute value, TemplateParamOp targetParam
) {
  // A wildcard `?` (represented as kDynamic) defers inference to a later pass. It is only valid
  // for parameters with a `!poly.tvar` type restriction.
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(value)) {
    if (isDynamic(intAttr)) {
      std::optional<Type> declaredType = targetParam.getTypeOpt();
      if (!declaredType || !llvm::isa<TypeVarType>(*declaredType)) {
        auto diag = origin->emitOpError().append(
            "wildcard `?` can only be used for template parameters with `!poly.tvar` "
            "type restriction, but parameter \"@",
            targetParam.getName(), "\" has "
        );
        if (declaredType) {
          diag.append("type restriction ", *declaredType);
        } else {
          diag.append("no type restriction");
        }
        return diag;
      }
      return success();
    }
  }

  std::optional<Type> declaredType = targetParam.getTypeOpt();
  bool compatible = !declaredType;
  if (auto sym = llvm::dyn_cast<SymbolRefAttr>(value)) {
    bool resolvedLocal = false;
    if (sym.getNestedReferences().empty()) {
      SymbolTableCollection tables;
      FailureOr<TemplateOp> parentTemplate = getConstResolutionTemplate(tables, origin);
      if (failed(parentTemplate)) {
        return failure();
      }
      if (TemplateOp p = *parentTemplate) {
        auto binding = p.getConstNamed<TemplateSymbolBindingOpInterface>(sym.getRootReference());
        if (binding) {
          resolvedLocal = true;
          if (declaredType) {
            compatible = isTemplateParamTypeCompatible(binding.getTypeOpt(), *declaredType);
          }
        }
      }
    }
    if (!resolvedLocal) {
      SymbolTableCollection tables;
      auto lookup = lookupTopLevelSymbol(tables, sym, origin);
      if (failed(lookup)) {
        return failure();
      }
      auto global = llvm::dyn_cast<GlobalDefOp>(lookup->get());
      if (!global) {
        return origin->emitOpError().append(
            "instantiation value '", value, "' refers to a '", lookup->get()->getName(),
            "' which is not allowed"
        );
      }
      if (!global.isConstant()) {
        auto diag = origin->emitOpError().append(
            "instantiation value '", value, "' refers to a global that is not marked as 'const'"
        );
        diag.attachNote(global.getLoc()).append("global defined here");
        return diag;
      }
      if (declaredType) {
        compatible = isTemplateParamTypeCompatible(global.getType(), *declaredType);
      }
    }
  } else if (declaredType && llvm::isa<TypeVarType>(*declaredType)) {
    TypeAttr typeValue = llvm::dyn_cast<TypeAttr>(value);
    compatible = static_cast<bool>(typeValue);
    if (typeValue) {
      if (failed(checkValidType(getEmitOpErrFn(origin), typeValue.getValue()))) {
        return failure();
      }
      // Resolve nested symbols now, while a valid TypeVarType remains deferred for inference.
      SymbolTableCollection tables;
      if (failed(verifyTypeResolution(tables, origin, typeValue.getValue()))) {
        return failure();
      }
    }
  } else if (declaredType) {
    compatible = succeeded(materializeTemplateParamValue(value, declaredType));
  }

  if (declaredType && !compatible) {
    return origin->emitOpError().append(
        "instantiation value '", value, "' is not compatible with parameter \"@",
        targetParam.getName(), "\" type restriction ", *declaredType
    );
  }
  return success();
}

LogicalResult verifyTemplateParamValuesCompatibility(
    Operation *origin, ArrayAttr explicitParams,
    llvm::iterator_range<Region::op_iterator<TemplateParamOp>> targetParamDefs
) {
  assert(!isNullOrEmpty(explicitParams) && "pre-condition");
  assert((explicitParams.size() == llvm::range_size(targetParamDefs)) && "pre-condition");

  for (auto [paramOp, attr] : llvm::zip_equal(targetParamDefs, explicitParams.getValue())) {
    // Affine maps are deferred within parameterized types, where a later operation supplies their
    // operands. Direct function and contract template-argument lists have no affine-map operands,
    // so index and integer restrictions require integer arguments.
    std::optional<Type> restriction = paramOp.getTypeOpt();
    if (llvm::isa<AffineMapAttr>(attr) && restriction &&
        llvm::isa<IndexType, IntegerType>(*restriction)) {
      return origin->emitOpError().append(
          "instantiation value '", attr, "' is not compatible with parameter \"@",
          paramOp.getName(), "\" type restriction ", *restriction
      );
    }
    if (failed(verifyTemplateParamValueCompatibility(origin, attr, paramOp))) {
      return failure();
    }
  }
  return success();
}

LogicalResult verifyKnownTargetTemplateParams(
    Operation *origin, FunctionType targetType, StringRef targetName, StringRef targetTemplateName,
    ArrayAttr explicitParams,
    llvm::iterator_range<Region::op_iterator<TemplateParamOp>> targetParamDefs,
    TemplateParamSignatureKind signatureKind,
    llvm::function_ref<FailureOr<UnificationMap>(UnificationCandidateFn)> unify
) {
  using CandidateMap =
      llvm::DenseMap<std::pair<SymbolRefAttr, Side>, llvm::SmallVector<Attribute, 2>>;
  CandidateMap candidateValues;
  auto recordCandidate = [&](SymbolRefAttr symbol, Side side, Attribute value) {
    auto &values = candidateValues[{symbol, side}];
    if (llvm::find(values, value) == values.end()) {
      values.push_back(value);
    }
  };
  auto getCandidates = [&](SymbolRefAttr symbol, Side side) -> ArrayRef<Attribute> {
    auto it = candidateValues.find({symbol, side});
    return it == candidateValues.end() ? ArrayRef<Attribute>() : it->second;
  };
  UnificationCandidateFn candidateRecorder = recordCandidate;

  StringRef signatureDescription = [&] {
    switch (signatureKind) {
    case TemplateParamSignatureKind::Function:
      return StringRef("function");
    case TemplateParamSignatureKind::Contract:
      return StringRef("contract");
    }
    llvm_unreachable("unknown template parameter signature kind");
  }();

  if (isNullOrEmpty(explicitParams)) {
    // Omitted arguments are valid only when every target parameter is exposed by the signature.
    llvm::SmallDenseSet<SymbolRefAttr> referencedInSignature;
    getSymbolsUsedIn(targetType.getInputs(), referencedInSignature);
    getSymbolsUsedIn(targetType.getResults(), referencedInSignature);

    bool allParamsReferenced = llvm::all_of(targetParamDefs, [&](TemplateParamOp param) {
      return referencedInSignature.contains(FlatSymbolRefAttr::get(param.getNameAttr()));
    });
    if (allParamsReferenced) {
      FailureOr<UnificationMap> unifyResult = unify(candidateRecorder);
      if (failed(unifyResult)) {
        return failure();
      }
      return verifyTemplateParamsMatchInferred(
          origin, explicitParams, targetParamDefs, unifyResult.value(), signatureKind, getCandidates
      );
    }
    return origin->emitOpError().append(
        "must provide template instantiation parameters when calling \"@", targetName,
        "\" because not all template parameters of \"@", targetTemplateName, "\" appear in the ",
        signatureDescription, " type signature"
    );
  }

  // Check that integer attributes can be represented as index values before validating them.
  if (failed(forceIntAttrTypes(explicitParams.getValue(), [origin] {
    return InFlightDiagnosticWrapper(origin->emitOpError());
  }))) {
    return failure();
  }

  size_t numTemplateParams = llvm::range_size(targetParamDefs);
  if (explicitParams.size() != numTemplateParams) {
    return origin->emitOpError().append(
        "template instantiation has ", explicitParams.size(), " parameter(s) but \"@",
        targetTemplateName, "\" expects ", numTemplateParams, " template parameter(s)"
    );
  }

  if (failed(verifyTemplateParamValuesCompatibility(origin, explicitParams, targetParamDefs))) {
    return failure();
  }

  // Compare explicit values with the target signature after local compatibility succeeds.
  FailureOr<UnificationMap> unifyResult = unify(candidateRecorder);
  if (failed(unifyResult)) {
    return failure();
  }
  return verifyTemplateParamsMatchInferred(
      origin, explicitParams, targetParamDefs, unifyResult.value(), signatureKind, getCandidates
  );
}

LogicalResult verifyTemplateParamsMatchInferred(
    Operation *origin, ArrayAttr explicitParams,
    llvm::iterator_range<Region::op_iterator<TemplateParamOp>> targetParamDefs,
    const UnificationMap &unifications, TemplateParamSignatureKind signatureKind,
    llvm::function_ref<ArrayRef<Attribute>(SymbolRefAttr, Side)> candidates
) {
  StringRef signatureDescription = [&] {
    switch (signatureKind) {
    case TemplateParamSignatureKind::Function:
      return StringRef("function");
    case TemplateParamSignatureKind::Contract:
      return StringRef("contract");
    }
    llvm_unreachable("unknown template parameter signature kind");
  }();

  if (isNullOrEmpty(explicitParams)) {
    for (TemplateParamOp paramOp : targetParamDefs) {
      FlatSymbolRefAttr paramName = FlatSymbolRefAttr::get(paramOp.getNameAttr());
      auto it = unifications.find({paramName, Side::RHS});
      if (it == unifications.end()) {
        return origin->emitOpError().append(
            "cannot infer template instantiation value for parameter \"@", paramOp.getName(),
            "\" from ", signatureDescription, " type signature"
        );
      }
      ArrayRef<Attribute> inferredCandidates =
          candidates ? candidates(paramName, Side::RHS) : ArrayRef<Attribute>();
      std::optional<Type> requiredType = paramOp.getTypeOpt();
      if (inferredCandidates.size() > 1 && requiredType &&
          llvm::isa<felt::FeltType>(*requiredType)) {
        if (failed(verifyRepeatedFeltCandidates(
                origin, paramOp, inferredCandidates, signatureDescription
            ))) {
          return failure();
        }
        continue;
      }
      if (!it->second) {
        return origin->emitOpError().append(
            "cannot infer template instantiation value for parameter \"@", paramOp.getName(),
            "\" from ", signatureDescription, " type signature"
        );
      }
      if (failed(verifyTemplateParamValueCompatibility(origin, it->second, paramOp))) {
        return failure();
      }
    }
    return success();
  }

  assert(!isNullOrEmpty(explicitParams) && "pre-condition");
  assert((explicitParams.size() == llvm::range_size(targetParamDefs)) && "pre-condition");

  for (auto [paramOp, attr] : llvm::zip_equal(targetParamDefs, explicitParams.getValue())) {
    // Skip wildcards (`?` / kDynamic) - their value will be resolved by a later inference pass.
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
      if (isDynamic(intAttr)) {
        continue;
      }
    }
    FlatSymbolRefAttr paramName = FlatSymbolRefAttr::get(paramOp.getNameAttr());
    auto it = unifications.find({paramName, Side::RHS});
    if (it != unifications.end() && !it->second) {
      ArrayRef<Attribute> inferredCandidates =
          candidates ? candidates(paramName, Side::RHS) : ArrayRef<Attribute>();
      std::optional<Type> requiredType = paramOp.getTypeOpt();
      if (inferredCandidates.size() > 1 && requiredType &&
          llvm::isa<felt::FeltType>(*requiredType)) {
        if (failed(verifyRepeatedFeltCandidates(
                origin, paramOp, inferredCandidates, signatureDescription, attr
            ))) {
          return failure();
        }
        continue;
      }
      return origin->emitOpError().append(
          "cannot infer a unique template instantiation value for parameter \"@", paramOp.getName(),
          "\" from ", signatureDescription, " type signature"
      );
    }
    if (it != unifications.end() &&
        failed(verifyTemplateParamValueCompatibility(origin, it->second, paramOp))) {
      return failure();
    }
    bool valuesUnify = true;
    if (it != unifications.end()) {
      SymbolTableCollection tables;
      FailureOr<bool> resolved =
          resolvedTemplateParamValuesUnify(tables, origin, attr, it->second, paramOp.getTypeOpt());
      if (failed(resolved)) {
        return failure();
      }
      valuesUnify = *resolved;
    }
    if (!valuesUnify) {
      return origin->emitOpError().append(
          "template instantiation value '", attr, "' for parameter \"@", paramOp.getName(),
          "\" conflicts with value '", it->second, "' inferred from ", signatureDescription,
          " type signature"
      );
    }
  }
  return success();
}

LogicalResult verifyParamOfType(
    SymbolTableCollection &tables, SymbolRefAttr param, Type parameterizedType, Operation *origin,
    std::optional<Type> requiredParamType, std::optional<Location> requiredParamLoc
) {
  // Most often, StructType and ArrayType SymbolRefAttr parameters will be defined as parameters of
  // the template that the current Operation is nested within. These are always flat references
  // (i.e., contain no nested references).
  if (param.getNestedReferences().empty()) {
    FailureOr<TemplateOp> parent = getConstResolutionTemplate(tables, origin);
    if (failed(parent)) {
      return failure(); // getConstResolutionTemplate() failure cases emit a sufficient error
                        // message
    }
    if (*parent) {
      if (auto b =
              parent->getConstNamed<TemplateSymbolBindingOpInterface>(param.getRootReference())) {
        return verifyTemplateSymbolType(
            b, param, parameterizedType, origin, requiredParamType, requiredParamLoc
        );
      }
    }
  }
  // Otherwise, see if the symbol can be found via lookup from the `origin` Operation.
  auto lookupRes = lookupTopLevelSymbol(tables, param, origin);
  if (failed(lookupRes)) {
    return failure(); // lookupTopLevelSymbol() already emits a sufficient error message
  }
  Operation *foundOp = lookupRes->get();
  auto global = llvm::dyn_cast<GlobalDefOp>(foundOp);
  if (!global) {
    return origin->emitError() << "ref \"" << param << "\" in type " << parameterizedType
                               << " refers to a '" << foundOp->getName()
                               << "' which is not allowed";
  }
  if (!global.isConstant()) {
    auto diag = origin->emitError() << "ref \"" << param << "\" in type " << parameterizedType
                                    << " refers to a global that is not marked as 'const'";
    diag.attachNote(global.getLoc()).append("global defined here");
    if (requiredParamLoc) {
      diag.attachNote(*requiredParamLoc).append("required parameter declared here");
    }
    return diag;
  }
  if (requiredParamType && !isTemplateParamTypeCompatible(global.getType(), *requiredParamType)) {
    auto diag = origin->emitError() << "ref \"" << param << "\" in type " << parameterizedType
                                    << " refers to a global with type " << global.getType()
                                    << " but expected " << *requiredParamType;
    diag.attachNote(global.getLoc()).append("global defined here");
    if (requiredParamLoc) {
      diag.attachNote(*requiredParamLoc).append("required parameter declared here");
    }
    return diag;
  }
  return success();
}

LogicalResult verifyParamsOfType(
    SymbolTableCollection &tables, ArrayRef<Attribute> tyParams, Type parameterizedType,
    Operation *origin, std::optional<Type> requiredParamType
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
      auto r = verifyParamOfType(tables, symRefParam, parameterizedType, origin, requiredParamType);
      if (failed(r)) {
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

namespace {

/// Type and value facts established by resolving one symbolic template argument.
struct TemplateParamSymbolEvidence {
  std::optional<Type> restriction;
  Attribute concreteValue;
};

/// Resolve a local template binding or qualified global without rejecting genuinely unknown refs.
FailureOr<std::optional<TemplateParamSymbolEvidence>> resolveTemplateParamSymbolEvidence(
    SymbolTableCollection &tables, Operation *origin, SymbolRefAttr symbol
) {
  if (symbol.getNestedReferences().empty()) {
    FailureOr<TemplateOp> parent = getConstResolutionTemplate(tables, origin);
    if (failed(parent)) {
      return failure();
    }
    if (*parent) {
      auto binding =
          parent->getConstNamed<TemplateSymbolBindingOpInterface>(symbol.getRootReference());
      if (binding) {
        return std::make_optional(TemplateParamSymbolEvidence {binding.getTypeOpt(), Attribute()});
      }
    }
  }

  auto global = lookupTopLevelSymbol<GlobalDefOp>(tables, symbol, origin, false);
  if (succeeded(global)) {
    GlobalDefOp globalOp = global->get();
    if (!globalOp.isConstant()) {
      return origin->emitError() << "template parameter symbol \"" << symbol
                                 << "\" refers to a global that is not marked as 'const'";
    }
    return std::make_optional(
        TemplateParamSymbolEvidence {
            globalOp.getType(),
            globalOp.getInitialValueAttr(),
        }
    );
  }
  return std::optional<TemplateParamSymbolEvidence>();
}

/// Return whether two known felt restrictions require different explicit fields.
bool feltRestrictionsConflict(std::optional<Type> lhs, std::optional<Type> rhs) {
  if (!lhs || !rhs) {
    return false;
  }
  auto lhsFelt = llvm::dyn_cast<felt::FeltType>(*lhs);
  auto rhsFelt = llvm::dyn_cast<felt::FeltType>(*rhs);
  return lhsFelt && rhsFelt && lhsFelt.hasField() && rhsFelt.hasField() && lhsFelt != rhsFelt;
}

/// Compare explicit and signature-inferred template values. For a felt restriction, local template
/// bindings contribute type evidence and qualified globals contribute type and concrete-value
/// evidence. Return `false` for a known field or value conflict, or when contextual materialization
/// rejects a value. Preserve the context-free unifier's result when either symbol has no resolvable
/// evidence. Return failure when the enclosing template scope cannot be resolved or a resolved
/// global is mutable. Non-felt restrictions always use the context-free unifier.
FailureOr<bool> resolvedTemplateParamValuesUnify(
    SymbolTableCollection &tables, Operation *origin, Attribute explicitValue,
    Attribute inferredValue, std::optional<Type> requiredParamType
) {
  bool contextFreeResult =
      templateParamValuesUnify(explicitValue, inferredValue, requiredParamType);
  if (!requiredParamType || !llvm::isa<felt::FeltType>(*requiredParamType)) {
    return contextFreeResult;
  }

  SymbolRefAttr explicitSymbol = llvm::dyn_cast<SymbolRefAttr>(explicitValue);
  SymbolRefAttr inferredSymbol = llvm::dyn_cast<SymbolRefAttr>(inferredValue);
  if (!explicitSymbol && !inferredSymbol) {
    return contextFreeResult;
  }

  std::optional<TemplateParamSymbolEvidence> explicitEvidence;
  std::optional<TemplateParamSymbolEvidence> inferredEvidence;
  if (explicitSymbol) {
    FailureOr<std::optional<TemplateParamSymbolEvidence>> resolved =
        resolveTemplateParamSymbolEvidence(tables, origin, explicitSymbol);
    if (failed(resolved)) {
      return failure();
    }
    explicitEvidence = *resolved;
  }
  if (inferredSymbol) {
    FailureOr<std::optional<TemplateParamSymbolEvidence>> resolved =
        resolveTemplateParamSymbolEvidence(tables, origin, inferredSymbol);
    if (failed(resolved)) {
      return failure();
    }
    inferredEvidence = *resolved;
  }

  // Unresolved references retain the generic unifier's deferral rule.
  if ((explicitSymbol && !explicitEvidence) || (inferredSymbol && !inferredEvidence)) {
    return contextFreeResult;
  }
  if (explicitEvidence && inferredEvidence &&
      feltRestrictionsConflict(explicitEvidence->restriction, inferredEvidence->restriction)) {
    return false;
  }

  // Replace a resolved global with its value; local bindings retain only their type evidence.
  auto materializeEvidence = [](
                                 Attribute fallback, SymbolRefAttr symbol,
                                 const std::optional<TemplateParamSymbolEvidence> &evidence
                             ) -> FailureOr<std::optional<Attribute>> {
    if (!symbol) {
      return std::make_optional(fallback);
    }
    if (!evidence || !evidence->concreteValue) {
      return std::optional<Attribute>();
    }
    FailureOr<Attribute> materialized =
        materializeTemplateParamValue(evidence->concreteValue, evidence->restriction);
    if (failed(materialized)) {
      return failure();
    }
    return std::make_optional(*materialized);
  };

  FailureOr<std::optional<Attribute>> explicitConcrete =
      materializeEvidence(explicitValue, explicitSymbol, explicitEvidence);
  FailureOr<std::optional<Attribute>> inferredConcrete =
      materializeEvidence(inferredValue, inferredSymbol, inferredEvidence);
  if (failed(explicitConcrete) || failed(inferredConcrete)) {
    return false;
  }
  if (*explicitConcrete && *inferredConcrete) {
    return templateParamValuesUnify(
        explicitConcrete->value(), inferredConcrete->value(), requiredParamType
    );
  }
  if (*explicitConcrete && inferredEvidence && inferredEvidence->restriction) {
    return succeeded(
        materializeTemplateParamValue(explicitConcrete->value(), inferredEvidence->restriction)
    );
  }
  if (*inferredConcrete && explicitEvidence && explicitEvidence->restriction) {
    return succeeded(
        materializeTemplateParamValue(inferredConcrete->value(), explicitEvidence->restriction)
    );
  }
  return contextFreeResult;
}

} // namespace

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
    if (TemplateOp parent = getParentOfType<TemplateOp>(defForType.getOperation())) {
      for (auto [paramOp, value] :
           llvm::zip_equal(parent.getConstOps<TemplateParamOp>(), tyParams.getValue())) {
        std::optional<Type> restriction = paramOp.getTypeOpt();
        if (auto symbolValue = llvm::dyn_cast<SymbolRefAttr>(value);
            symbolValue && restriction &&
            failed(
                verifyParamOfType(tables, symbolValue, ty, origin, restriction, paramOp.getLoc())
            )) {
          return failure();
        }
      }
    }
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
    auto r = verifyParamsOfType(
        tables, aTy.getDimensionSizes(), aTy, origin, IndexType::get(aTy.getContext())
    );
    if (failed(r)) {
      return failure();
    }
    return verifyTypeResolution(tables, origin, aTy.getElementType());
  } else if (TypeVarType vTy = llvm::dyn_cast<TypeVarType>(ty)) {
    // Unlike other type parameters, a type variable may only name a parameter
    // of the enclosing template; it cannot resolve to a global.
    FailureOr<TemplateOp> parent = getConstResolutionTemplate(tables, origin);
    if (failed(parent)) {
      return failure();
    }
    TemplateOp templateOp = *parent;
    if (templateOp && templateOp.getConstNamed<TemplateParamOp>(vTy.getNameRef())) {
      return success();
    }
    return origin->emitError() << "type variable " << vTy
                               << " must reference a parameter of its enclosing " << '\''
                               << TemplateOp::getOperationName() << '\'';
  } else {
    return success();
  }
}

} // namespace llzk
