//===-- SymbolTableLLZK.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from the LLVM Project's mlir/lib/IR/SymbolTable.cpp
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This is a selection of code from `mlir/lib/IR/SymbolTable.cpp` to support a
/// modified version of `walkSymbolRefs()` so the "symbol use" functions will
/// also consider symbols used within the operand and result types of Ops. The
/// problem is that `walkSymbolRefs()` only searches `op->getAttrDictionary()`
/// which does not include any Type instances used on the operand and result SSA
/// values but in LLZK, the StructType can contain symbol references so they
/// should be included in the results here. Only `walkSymbolRefs()` and
/// `getSymbolName()` are functionally modified from their MLIR versions.
///
/// An alternative solution that could be explored further is adding an
/// additional kind of Attribute on all ops that ensures the operand and result
/// Type instances are included in the walk. However, that approach is not
/// straightforward to implement. When would these be added? How would they be
/// kept in sync with the actual Values? MLIR allows mutable attributes/types
/// but it doesn't seem like the walk would pick up whatever's included in the
/// mutable part of the underlying storage because that part cannot be included
/// in the storage `KeyTy` and IIUC, the walk recursion is based on the `KeyTy`.
///
//===----------------------------------------------------------------------===//

#include "llzk/Util/SymbolTableLLZK.h"

#include <llvm/ADT/SmallPtrSet.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Symbol Use Lists
//===----------------------------------------------------------------------===//

namespace {

/// Return true if the given operation is unknown and may potentially define a
/// symbol table.
static bool isPotentiallyUnknownSymbolTable(Operation *op) {
  return op->getNumRegions() == 1 && !op->getDialect();
}

/// Returns the string name of the given symbol, or null if this is not a symbol.
static StringAttr getNameIfSymbol(Operation *op, StringAttr symbolAttrNameId) {
  return op->getAttrOfType<StringAttr>(symbolAttrNameId);
}

/// Computes the nested symbol reference attribute for the symbol 'symbolName'
/// that are usable within the symbol table operations from 'symbol' as far up
/// to the given operation 'within', where 'within' is an ancestor of 'symbol'.
/// Returns success if all references up to 'within' could be computed.
static LogicalResult collectValidReferencesFor(
    Operation *symbol, StringAttr symbolName, Operation *within,
    SmallVectorImpl<SymbolRefAttr> &results
) {
  assert(within->isAncestor(symbol) && "expected 'within' to be an ancestor");
  MLIRContext *ctx = symbol->getContext();

  auto leafRef = FlatSymbolRefAttr::get(symbolName);
  results.push_back(leafRef);

  // Early exit for when 'within' is the parent of 'symbol'.
  Operation *symbolTableOp = symbol->getParentOp();
  if (within == symbolTableOp) {
    return success();
  }

  // Collect references until 'symbolTableOp' reaches 'within'.
  SmallVector<FlatSymbolRefAttr, 1> nestedRefs(1, leafRef);
  StringAttr symbolNameId = StringAttr::get(ctx, SymbolTable::getSymbolAttrName());
  do {
    // Each parent of 'symbol' should define a symbol table.
    if (!symbolTableOp->hasTrait<OpTrait::SymbolTable>()) {
      return failure();
    }
    // Each parent of 'symbol' should also be a symbol.
    StringAttr symbolTableName = getNameIfSymbol(symbolTableOp, symbolNameId);
    if (!symbolTableName) {
      return failure();
    }
    results.push_back(SymbolRefAttr::get(symbolTableName, nestedRefs));

    symbolTableOp = symbolTableOp->getParentOp();
    if (symbolTableOp == within) {
      break;
    }
    nestedRefs.insert(nestedRefs.begin(), FlatSymbolRefAttr::get(symbolTableName));
  } while (true);
  return success();
}

/// Walk all of the operations within the given set of regions, without
/// traversing into any nested symbol tables. Stops walking if the result of the
/// callback is anything other than `WalkResult::advance`.
static std::optional<WalkResult> walkSymbolTable(
    MutableArrayRef<Region> regions, function_ref<std::optional<WalkResult>(Operation *)> callback
) {
  SmallVector<Region *, 1> worklist(llvm::make_pointer_range(regions));
  while (!worklist.empty()) {
    for (Operation &op : worklist.pop_back_val()->getOps()) {
      std::optional<WalkResult> result = callback(&op);
      if (result != WalkResult::advance()) {
        return result;
      }

      // If this op defines a new symbol table scope, we can't traverse. Any
      // symbol references nested within 'op' are different semantically.
      if (!op.hasTrait<OpTrait::SymbolTable>()) {
        for (Region &region : op.getRegions()) {
          worklist.push_back(&region);
        }
      }
    }
  }
  return WalkResult::advance();
}

/// Walk all of the symbol references within the given operation, invoking the provided
/// `callback` for each found use. The `callback` takes the use of the symbol as input.
static WalkResult
walkSymbolRefs(Operation *op, function_ref<WalkResult(SymbolTable::SymbolUse)> callback) {
  // This is modified for LLZK.
  auto walkFn = [&op, &callback](SymbolRefAttr symbolRef) {
    if (callback({op, symbolRef}).wasInterrupted()) {
      return WalkResult::interrupt();
    }
    return WalkResult::skip(); // Don't walk nested references.
  };
  for (Type t : op->getOperandTypes()) {
    if (t.walk<WalkOrder::PreOrder>(walkFn).wasInterrupted()) {
      return WalkResult::interrupt();
    }
  }
  for (Type t : op->getResultTypes()) {
    if (t.walk<WalkOrder::PreOrder>(walkFn).wasInterrupted()) {
      return WalkResult::interrupt();
    }
  }
  return op->getAttrDictionary().walk<WalkOrder::PreOrder>(walkFn);
}

/// Walk all of the uses, for any symbol, that are nested within the given
/// regions, invoking the provided callback for each. This does not traverse
/// into any nested symbol tables.
static std::optional<WalkResult> walkSymbolUses(
    MutableArrayRef<Region> regions, function_ref<WalkResult(SymbolTable::SymbolUse)> callback
) {
  return walkSymbolTable(regions, [&](Operation *op) -> std::optional<WalkResult> {
    // Check that this isn't a potentially unknown symbol table.
    if (isPotentiallyUnknownSymbolTable(op)) {
      return std::nullopt;
    }
    return walkSymbolRefs(op, callback);
  });
}

/// Walk all of the uses, for any symbol, that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables.
static std::optional<WalkResult>
walkSymbolUses(Operation *from, function_ref<WalkResult(SymbolTable::SymbolUse)> callback) {
  // If this operation has regions, and it, as well as its dialect, isn't
  // registered then conservatively fail. The operation may define a
  // symbol table, so we can't opaquely know if we should traverse to find
  // nested uses.
  if (isPotentiallyUnknownSymbolTable(from)) {
    return std::nullopt;
  }

  // Walk the uses on this operation.
  if (walkSymbolRefs(from, callback).wasInterrupted()) {
    return WalkResult::interrupt();
  }

  // Only recurse if this operation is not a symbol table. A symbol table
  // defines a new scope, so we can't walk the attributes from within the symbol
  // table op.
  if (!from->hasTrait<OpTrait::SymbolTable>()) {
    return walkSymbolUses(from->getRegions(), callback);
  }
  return WalkResult::advance();
}

/// This class represents a single symbol scope. A symbol scope represents the
/// set of operations nested within a symbol table that may reference symbols
/// within that table. A symbol scope does not contain the symbol table
/// operation itself, just its contained operations. A scope ends at leaf
/// operations or another symbol table operation.
struct SymbolScope {
  /// Walk the symbol uses within this scope, invoking the given callback.
  /// This variant is used when the callback type matches that expected by
  /// 'walkSymbolUses'.
  template <
      typename CallbackT,
      std::enable_if_t<!std::is_same<
          typename llvm::function_traits<CallbackT>::result_t, void>::value> * = nullptr>
  std::optional<WalkResult> walk(CallbackT cback) {
    if (Region *region = llvm::dyn_cast_if_present<Region *>(limit)) {
      return walkSymbolUses(*region, cback);
    }
    return walkSymbolUses(llvm::cast<Operation *>(limit), cback);
  }
  /// This variant is used when the callback type matches a stripped down type:
  /// void(SymbolTable::SymbolUse use)
  template <
      typename CallbackT,
      std::enable_if_t<std::is_same<
          typename llvm::function_traits<CallbackT>::result_t, void>::value> * = nullptr>
  std::optional<WalkResult> walk(CallbackT cback) {
    return walk([=](SymbolTable::SymbolUse use) { return cback(use), WalkResult::advance(); });
  }

  /// Walk all of the operations nested under the current scope without
  /// traversing into any nested symbol tables.
  template <typename CallbackT> std::optional<WalkResult> walkSymbolTable(CallbackT &&cback) {
    if (Region *region = llvm::dyn_cast_if_present<Region *>(limit)) {
      return ::walkSymbolTable(*region, cback);
    }
    return ::walkSymbolTable(llvm::cast<Operation *>(limit), cback);
  }

  /// The representation of the symbol within this scope.
  SymbolRefAttr symbol;

  /// The IR unit representing this scope.
  llvm::PointerUnion<Operation *, Region *> limit;
};

/// Collect all of the symbol scopes from 'symbol' to (inclusive) 'limit'.
static SmallVector<SymbolScope, 2> collectSymbolScopes(Operation *symbol, Operation *limit) {
  StringAttr symName = SymbolTable::getSymbolName(symbol);
  assert(!symbol->hasTrait<OpTrait::SymbolTable>() || symbol != limit);

  // Compute the ancestors of 'limit'.
  SetVector<Operation *, SmallVector<Operation *, 4>, SmallPtrSet<Operation *, 4>> limitAncestors;
  Operation *limitAncestor = limit;
  do {
    // Check to see if 'symbol' is an ancestor of 'limit'.
    if (limitAncestor == symbol) {
      // Check that the nearest symbol table is 'symbol's parent. SymbolRefAttr
      // doesn't support parent references.
      if (SymbolTable::getNearestSymbolTable(limit->getParentOp()) == symbol->getParentOp()) {
        return {{SymbolRefAttr::get(symName), limit}};
      }
      return {};
    }

    limitAncestors.insert(limitAncestor);
  } while ((limitAncestor = limitAncestor->getParentOp()));

  // Try to find the first ancestor of 'symbol' that is an ancestor of 'limit'.
  Operation *commonAncestor = symbol->getParentOp();
  do {
    if (limitAncestors.count(commonAncestor)) {
      break;
    }
  } while ((commonAncestor = commonAncestor->getParentOp()));
  assert(commonAncestor && "'limit' and 'symbol' have no common ancestor");

  // Compute the set of valid nested references for 'symbol' as far up to the
  // common ancestor as possible.
  SmallVector<SymbolRefAttr, 2> references;
  bool collectedAllReferences =
      succeeded(collectValidReferencesFor(symbol, symName, commonAncestor, references));

  // Handle the case where the common ancestor is 'limit'.
  if (commonAncestor == limit) {
    SmallVector<SymbolScope, 2> scopes;

    // Walk each of the ancestors of 'symbol', calling the compute function for
    // each one.
    Operation *limitIt = symbol->getParentOp();
    for (size_t i = 0, e = references.size(); i != e; ++i, limitIt = limitIt->getParentOp()) {
      assert(limitIt->hasTrait<OpTrait::SymbolTable>());
      scopes.push_back({references[i], &limitIt->getRegion(0)});
    }
    return scopes;
  }

  // Otherwise, we just need the symbol reference for 'symbol' that will be
  // used within 'limit'. This is the last reference in the list we computed
  // above if we were able to collect all references.
  if (!collectedAllReferences) {
    return {};
  }
  return {{references.back(), limit}};
}

static SmallVector<SymbolScope, 2> collectSymbolScopes(Operation *symbol, Region *limit) {
  auto scopes = collectSymbolScopes(symbol, limit->getParentOp());

  // If we collected some scopes to walk, make sure to constrain the one for
  // limit to the specific region requested.
  if (!scopes.empty()) {
    scopes.back().limit = limit;
  }
  return scopes;
}

static SmallVector<SymbolScope, 1> collectSymbolScopes(StringAttr symbol, Region *limit) {
  return {{SymbolRefAttr::get(symbol), limit}};
}

static SmallVector<SymbolScope, 1> collectSymbolScopes(StringAttr symbol, Operation *limit) {
  SmallVector<SymbolScope, 1> scopes;
  auto symbolRef = SymbolRefAttr::get(symbol);
  for (auto &region : limit->getRegions()) {
    scopes.push_back({symbolRef, &region});
  }
  return scopes;
}

/// Returns true if the given reference 'SubRef' is a sub reference of the
/// reference 'ref', i.e., 'ref' is a further qualified reference.
static bool isReferencePrefixOf(SymbolRefAttr subRef, SymbolRefAttr ref) {
  if (ref == subRef) {
    return true;
  }

  // If the references are not pointer equal, check to see if `subRef` is a
  // prefix of `ref`.
  if (llvm::isa<FlatSymbolRefAttr>(ref) || ref.getRootReference() != subRef.getRootReference()) {
    return false;
  }

  auto refLeafs = ref.getNestedReferences();
  auto subRefLeafs = subRef.getNestedReferences();
  return subRefLeafs.size() < refLeafs.size() &&
         subRefLeafs == refLeafs.take_front(subRefLeafs.size());
}

} // namespace

//===----------------------------------------------------------------------===//
// llzk::getSymbolUses

namespace {

/// The implementation of llzk::getSymbolUses below.
template <typename FromT>
static std::optional<SymbolTable::UseRange> getSymbolUsesImpl(FromT from) {
  std::vector<SymbolTable::SymbolUse> uses;
  auto walkFn = [&](SymbolTable::SymbolUse symbolUse) {
    uses.push_back(symbolUse);
    return WalkResult::advance();
  };
  auto result = walkSymbolUses(from, walkFn);
  return result ? std::optional<SymbolTable::UseRange>(std::move(uses)) : std::nullopt;
}

} // namespace

/// Get an iterator range for all of the uses, for any symbol, that are nested
/// within the given operation 'from'. This does not traverse into any nested
/// symbol tables, and will also only return uses on 'from' if it does not
/// also define a symbol table. This is because we treat the region as the
/// boundary of the symbol table, and not the op itself. This function returns
/// std::nullopt if there are any unknown operations that may potentially be
/// symbol tables.
std::optional<SymbolTable::UseRange> llzk::getSymbolUses(Operation *from) {
  return getSymbolUsesImpl(from);
}
std::optional<SymbolTable::UseRange> llzk::getSymbolUses(Region *from) {
  return getSymbolUsesImpl(MutableArrayRef<Region>(*from));
}

//===----------------------------------------------------------------------===//
// llzk::getSymbolUses

namespace {

/// The implementation of llzk::getSymbolUses below.
template <typename SymbolT, typename IRUnitT>
static std::optional<SymbolTable::UseRange> getSymbolUsesImpl(SymbolT symbol, IRUnitT *limit) {
  std::vector<SymbolTable::SymbolUse> uses;
  for (SymbolScope &scope : collectSymbolScopes(symbol, limit)) {
    if (!scope.walk([&](SymbolTable::SymbolUse symbolUse) {
      if (isReferencePrefixOf(scope.symbol, symbolUse.getSymbolRef())) {
        uses.push_back(symbolUse);
      }
    })) {
      return std::nullopt;
    }
  }
  return SymbolTable::UseRange(std::move(uses));
}

} // namespace

/// Get all of the uses of the given symbol that are nested within the given
/// operation 'from', invoking the provided callback for each. This does not
/// traverse into any nested symbol tables. This function returns std::nullopt
/// if there are any unknown operations that may potentially be symbol tables.
std::optional<SymbolTable::UseRange> llzk::getSymbolUses(StringAttr symbol, Operation *from) {
  return getSymbolUsesImpl(symbol, from);
}
std::optional<SymbolTable::UseRange> llzk::getSymbolUses(Operation *symbol, Operation *from) {
  return getSymbolUsesImpl(symbol, from);
}
std::optional<SymbolTable::UseRange> llzk::getSymbolUses(StringAttr symbol, Region *from) {
  return getSymbolUsesImpl(symbol, from);
}
std::optional<SymbolTable::UseRange> llzk::getSymbolUses(Operation *symbol, Region *from) {
  return getSymbolUsesImpl(symbol, from);
}

//===----------------------------------------------------------------------===//
// llzk::getSymbolsUsedIn

void llzk::getSymbolsUsedIn(Type t, llvm::SmallDenseSet<SymbolRefAttr> &symbolsUsed) {
  t.walk([&symbolsUsed](SymbolRefAttr symbolRef) { symbolsUsed.insert(symbolRef); });
}

void llzk::getSymbolsUsedIn(ArrayRef<Type> types, llvm::SmallDenseSet<SymbolRefAttr> &symbolsUsed) {
  for (Type t : types) {
    getSymbolsUsedIn(t, symbolsUsed);
  }
}

llvm::SmallDenseSet<SymbolRefAttr> llzk::getSymbolsUsedIn(Type t) {
  llvm::SmallDenseSet<SymbolRefAttr> symbolsUsed;
  getSymbolsUsedIn(t, symbolsUsed);
  return symbolsUsed;
}

llvm::SmallDenseSet<SymbolRefAttr> llzk::getSymbolsUsedIn(ArrayRef<Type> types) {
  llvm::SmallDenseSet<SymbolRefAttr> symbolsUsed;
  getSymbolsUsedIn(types, symbolsUsed);
  return symbolsUsed;
}

//===----------------------------------------------------------------------===//
// llzk::symbolKnownUseEmpty

namespace {

/// The implementation of llzk::symbolKnownUseEmpty below.
template <typename SymbolT, typename IRUnitT>
static bool symbolKnownUseEmptyImpl(SymbolT symbol, IRUnitT *limit) {
  for (SymbolScope &scope : collectSymbolScopes(symbol, limit)) {
    // Walk all of the symbol uses looking for a reference to 'symbol'.
    if (scope.walk([&](SymbolTable::SymbolUse symbolUse) {
      return isReferencePrefixOf(scope.symbol, symbolUse.getSymbolRef()) ? WalkResult::interrupt()
                                                                         : WalkResult::advance();
    }) != WalkResult::advance()) {
      return false;
    }
  }
  return true;
}

} // namespace

/// Return if the given symbol is known to have no uses that are nested within
/// the given operation 'from'. This does not traverse into any nested symbol
/// tables. This function will also return false if there are any unknown
/// operations that may potentially be symbol tables.
bool llzk::symbolKnownUseEmpty(StringAttr symbol, Operation *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}
bool llzk::symbolKnownUseEmpty(Operation *symbol, Operation *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}
bool llzk::symbolKnownUseEmpty(StringAttr symbol, Region *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}
bool llzk::symbolKnownUseEmpty(Operation *symbol, Region *from) {
  return symbolKnownUseEmptyImpl(symbol, from);
}

//===----------------------------------------------------------------------===//
// llzk::getSymbolName

StringAttr llzk::getSymbolName(Operation *op) {
  // This is modified for LLZK.
  // `SymbolTable::getSymbolName(Operation*)` asserts if there is no name (ex: in the case of
  // ModuleOp where the symbol name is optional) and there's no other way to check if the name
  // exists so this fully involved retrieval method must be used to return `nullptr` if no name.
  return op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
}
