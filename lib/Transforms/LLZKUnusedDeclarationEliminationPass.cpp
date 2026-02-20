//===-- LLZKUnusedDeclarationEliminationPass.cpp ----------------*- C++ -*-===//
//
// Part of the LLZK Project, under the Apache License v2.0.
// See LICENSE.txt for license information.
// Copyright 2025 Veridise Inc.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the `-llzk-unused-declaration-elim` pass.
///
//===----------------------------------------------------------------------===//

#include "llzk/Dialect/Struct/IR/Ops.h"
#include "llzk/Transforms/LLZKTransformationPasses.h"
#include "llzk/Util/SymbolHelper.h"
#include "llzk/Util/SymbolLookup.h"

#include <mlir/IR/BuiltinOps.h>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Debug.h>

// Include the generated base pass class definitions.
namespace llzk {
// the *DECL* macro is required when a pass has options to declare the option struct
#define GEN_PASS_DECL_UNUSEDDECLARATIONELIMINATIONPASS
#define GEN_PASS_DEF_UNUSEDDECLARATIONELIMINATIONPASS
#include "llzk/Transforms/LLZKTransformationPasses.h.inc"
} // namespace llzk

using namespace mlir;
using namespace llzk;
using namespace llzk::component;

#define DEBUG_TYPE "llzk-unused-declaration-elim"

namespace {

/// @brief Get the fully-qualified member symbol.
SymbolRefAttr getFullMemberSymbol(MemberRefOpInterface op) {
  SymbolRefAttr structSym = op.getStructType().getNameRef(); // this is fully qualified
  return appendLeaf(structSym, op.getMemberNameAttr());
}

class UnusedDeclarationEliminationPass
    : public llzk::impl::UnusedDeclarationEliminationPassBase<UnusedDeclarationEliminationPass> {

  /// @brief Shared context between the operations in this pass (member removal, struct removal)
  /// that doesn't need to be persisted after the pass completes.
  struct PassContext {
    DenseMap<SymbolRefAttr, StructDefOp> symbolToStruct;
    DenseMap<StructDefOp, SymbolRefAttr> structToSymbol;

    const SymbolRefAttr &getSymbol(StructDefOp s) const { return structToSymbol.at(s); }
    StructDefOp getStruct(const SymbolRefAttr &sym) const { return symbolToStruct.at(sym); }

    static PassContext populate(ModuleOp modOp) {
      PassContext ctx;

      modOp.walk<WalkOrder::PreOrder>([&ctx](StructDefOp structDef) {
        auto structSymbolRes = getPathFromTopRoot(structDef);
        ensure(succeeded(structSymbolRes), "failed to lookup struct symbol");
        SymbolRefAttr structSym = *structSymbolRes;
        ctx.symbolToStruct[structSym] = structDef;
        ctx.structToSymbol[structDef] = structSym;
      });
      return ctx;
    }
  };

  void runOnOperation() override {
    PassContext ctx = PassContext::populate(getOperation());
    // First, remove unused members. This may allow more structs to be removed,
    // if their final remaining uses are as types for unused members.
    removeUnusedMembers(ctx);

    // Last, remove unused structs if configured
    if (removeStructs) {
      removeUnusedStructs(ctx);
    }
  }

  /// @brief Removes unused members.
  /// A member is unused if it is never read from (only written to).
  /// @param structDef
  void removeUnusedMembers(PassContext &ctx) {
    ModuleOp modOp = getOperation();

    // Map fully-qualified member symbols -> member ops
    DenseMap<SymbolRefAttr, MemberDefOp> members;
    for (auto &[structDef, structSym] : ctx.structToSymbol) {
      structDef.walk([&](MemberDefOp member) {
        // We don't consider public members in the Main component for removal,
        // as these are output values and removing them would result in modifying
        // the overall circuit interface.
        if (!structDef.isMainComponent() || !member.hasPublicAttr()) {
          SymbolRefAttr memberSym =
              appendLeaf(structSym, FlatSymbolRefAttr::get(member.getSymNameAttr()));
          members[memberSym] = member;
        }
      });
    }

    // Remove all members that are read.
    modOp.walk([&](MemberReadOp readm) {
      SymbolRefAttr readMemberSym = getFullMemberSymbol(readm);
      members.erase(readMemberSym);
    });

    // Remove all writes that reference the remaining members, as these writes
    // are now known to only update write-only members.
    modOp.walk([&](MemberWriteOp writem) {
      SymbolRefAttr writtenMember = getFullMemberSymbol(writem);
      if (members.contains(writtenMember)) {
        // We need not check the users of a writem, since it produces no results.
        LLVM_DEBUG(
            llvm::dbgs() << "Removing write " << writem << " to write-only member " << writtenMember
                         << '\n'
        );
        writem.erase();
      }
    });

    // Finally, erase the remaining members.
    for (auto &[_, memberDef] : members) {
      LLVM_DEBUG(llvm::dbgs() << "Removing member " << memberDef << '\n');
      memberDef->erase();
    }
  }

  /// @brief Remove unused structs by looking for any uses of the struct's fully-qualified
  /// symbol. This catches any uses, such as member declarations of the struct's type
  /// or calls to any of the struct's methods.
  /// @param ctx
  void removeUnusedStructs(PassContext &ctx) {
    DenseMap<StructDefOp, DenseSet<StructDefOp>> uses;
    DenseMap<StructDefOp, DenseSet<StructDefOp>> usedBy;

    // initialize both maps with empty sets so we can identify unused structs
    for (auto &[structDef, _] : ctx.structToSymbol) {
      uses[structDef] = {};
      usedBy[structDef] = {};
    }

    getOperation().walk([&](Operation *op) {
      auto structParent = op->getParentOfType<StructDefOp>();
      if (structParent == nullptr) {
        return WalkResult::advance();
      }

      auto tryAddUse = [&](Type ty) {
        if (auto structTy = dyn_cast<StructType>(ty)) {
          // This name ref is required to be fully qualified
          SymbolRefAttr sym = structTy.getNameRef();
          StructDefOp refStruct = ctx.getStruct(sym);
          if (refStruct != structParent) {
            uses[structParent].insert(refStruct);
            usedBy[refStruct].insert(structParent);
          }
        }
      };

      // LLZK requires fully-qualified references to struct symbols. So, we
      // simply need to look for the struct symbol within this op's symbol uses.

      // Check operands
      for (Value operand : op->getOperands()) {
        tryAddUse(operand.getType());
      }

      // Check results
      for (Value result : op->getResults()) {
        tryAddUse(result.getType());
      }

      // Check block arguments
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument arg : block.getArguments()) {
            tryAddUse(arg.getType());
          }
        }
      }

      // Check attributes
      for (const auto &namedAttr : op->getAttrs()) {
        namedAttr.getValue().walk([&](TypeAttr typeAttr) { tryAddUse(typeAttr.getValue()); });
      }

      return WalkResult::advance();
    });

    SmallVector<StructDefOp> unusedStructs;

    auto updateUnusedStructs = [&]() {
      for (auto &[structDef, users] : usedBy) {
        if (users.empty() && !structDef.isMainComponent()) {
          unusedStructs.push_back(structDef);
        }
      }
    };

    updateUnusedStructs();

    while (!unusedStructs.empty()) {
      StructDefOp unusedStruct = unusedStructs.back();
      unusedStructs.pop_back();

      // See what structs are being used by this unused struct
      for (auto usedStruct : uses[unusedStruct]) {
        // The usedStruct is no longer used by the unusedStruct
        usedBy[usedStruct].erase(unusedStruct);
      }

      // Remove the unused struct from both maps and the IR
      usedBy.erase(unusedStruct);
      uses.erase(unusedStruct);
      unusedStruct->erase();

      // Check to see if we've created any more unused structs after we process
      // all existing known unused structs (to avoid double processing).
      if (unusedStructs.empty()) {
        updateUnusedStructs();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> llzk::createUnusedDeclarationEliminationPass() {
  return std::make_unique<UnusedDeclarationEliminationPass>();
};
